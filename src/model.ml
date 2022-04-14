(* train a RFC from a space-separated csv file (training set);
   first column must be the classification label (1: class of interest; -1: other class) *)

open Printf

module A = BatArray
module CLI = Minicli.CLI
module Fn = Filename
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module Rf = OrrandomForest.Rf
module S = BatString
module Utls = OrrandomForest.Utls

module Score_label = struct
  type t = bool * float (* (label, pred_score) *)
  let get_label (l, _) = l
  let get_score (_, s) = s
end

module ROC = Cpm.MakeROC.Make(Score_label)

let run_command (cmd: string): unit =
  Log.info "run_command: %s" cmd;
  match Unix.system cmd with
  | Unix.WSIGNALED _ -> (Log.fatal "run_command: signaled: %s" cmd; exit 1)
  | Unix.WSTOPPED _ -> (Log.fatal "run_command: stopped: %s" cmd; exit 1)
  | Unix.WEXITED i when i <> 0 ->
    (Log.fatal "run_command: exit %d: %s" i cmd; exit 1)
  | Unix.WEXITED _ (* i = 0 then *) -> ()

let train_classifier verbose nb_trees nb_features data_fn labels_fn =
  let params: Rf.params =
    { ntree = nb_trees;
      mtry = int_of_float (floor (sqrt (float nb_features)));
      importance = true } in
  Rf.(train ~debug:verbose
        Classification
        Rf.Dense
        params
        data_fn
        labels_fn)

let test_classifier verbose trained_model data_fn =
  Rf.(read_predictions
        (predict ~debug:verbose Classification Dense trained_model data_fn))

let split_label_features data_csv_fn =
  let tmp_labels_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_labels_" ".csv" in
  let tmp_features_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_features_" ".csv" in
  let cmd1 = sprintf "cut -d' ' -f1 %s > %s" data_csv_fn tmp_labels_fn in
  let cmd2 = sprintf "cut -d' ' -f2- %s | tr ' ' '\t' > %s" data_csv_fn tmp_features_fn in
  run_command cmd1;
  run_command cmd2;
  let labels = LO.lines_of_file tmp_labels_fn in
  (* labels must be tab-separated, on a single line *)
  LO.with_out_file tmp_labels_fn (fun out ->
      L.iteri (fun i label ->
          if i = 0 then
            fprintf out "%s" label
          else
            fprintf out "\t%s" label
        ) labels;
      fprintf out "\n"
    );
  (tmp_features_fn, tmp_labels_fn)

               (* [-np <int>]: max number of processes (default=1)\n  \
                * [-n <int>]: |RF|; default=100\n  \
                * [--mtry <float>]: proportion of randomly selected features\n  \
                * to use at each split (default=(sqrt(|features|))/|features|)\n  \
                * [--scan-mtry]: scan for best mtry in [0.001,0.002,0.005,...,1.0]\n  \
                * (incompatible with --mtry)\n  \
                * [--mtry-range <string>]: mtrys to test e.g. \"0.001,0.002,0.005\"\n  \
                * [-o <filename>]: output scores to file\n  \
                * [--NxCV <int>]: number of folds of cross validation\n  \
                * [--no-regr-plot]: turn OFF regression plot\n  \
                * [--rec-plot]: turn ON REC curve\n  \
                * [--y-rand]: turn ON Y-randomization\n  \
                * [-s <filename>]: save model to file\n  \
                * [-l <filename>]: load model from file\n  \
                * [--max-feat <int>]: max feature id.  \
                * (cf. end of encoding dict)\n  \
                * [-h|--help]: show this help message\n" *)

let main () =
  let argc, args = CLI.init () in
  Log.(set_log_level INFO);
  Log.color_on ();
  let train_portion_def = 0.8 in
  let nb_trees_def = 100 in
  if argc = 1 then
    begin
      eprintf "usage:\n\
               %s  \
               -i <train.csv>: training set\n  \
               [-p <float>]: proportion of the (randomized) dataset\n  \
               used to train (default=%.2f)\n  \
               [--seed <int>: fix random seed]\n  \
               [-n <int>]: num_trees=|RF|; default=%d\n  \
               [-v]: verbose/debug mode\n  \
              "
        Sys.argv.(0) train_portion_def nb_trees_def;
      exit 1
    end;
  let input_fn = CLI.get_string ["-i"] args in
  let p = CLI.get_float_def ["-p"] args train_portion_def in
  let rng = match CLI.get_int_opt ["--seed"] args with
    | None -> Random.State.make_self_init ()
    | Some seed -> Random.State.make [|seed|] in
  let nb_trees = CLI.get_int_def ["-n"] args nb_trees_def in
  let verbose = CLI.get_set_bool ["-v"] args in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let header, all_lines =
    let lines = LO.lines_of_file input_fn in
    match lines with
    | header' :: data_lines ->
      assert(S.starts_with header' "#");
      (S.lchop header', L.shuffle ~state:rng data_lines)
    | _ -> failwith ("not enough lines in: " ^ input_fn) in
  Log.info "header: %s" header;
  let n = L.length all_lines in
  let tmp_train_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_train_" ".csv" in
  let tmp_test_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_test_" ".csv" in
  let train_n, test_n =
    let x = BatFloat.round_to_int (p *. (float_of_int n)) in
    (x, n - x) in
  Log.info "train/test: %d/%d" train_n test_n;
  let train_lines, test_lines = L.takedrop train_n all_lines in
  LO.lines_to_file tmp_train_fn train_lines;
  LO.lines_to_file tmp_test_fn test_lines;
  let train_features_fn, train_labels_fn = split_label_features tmp_train_fn in
  let nb_features = S.count_char (L.hd all_lines) ' ' in
  Log.info "|features|=%d" nb_features;
  let model = train_classifier verbose nb_trees nb_features train_features_fn train_labels_fn in
  let feat_importance = Rf.read_predictions (Rf.get_features_importance model) in
  let index2feature_name = A.of_list (L.tl (S.split_on_char ' ' header)) in
  assert(L.length feat_importance = A.length index2feature_name);
  assert(A.length index2feature_name = nb_features);
  L.iteri (fun i imp ->
      Log.info "imp(%s): %.2f" index2feature_name.(i) imp
    ) feat_importance;
  let test_features_fn, test_labels_fn = split_label_features tmp_test_fn in
  let test_preds = test_classifier verbose model test_features_fn in
  let test_labels =
    (* all labels are on a single line in the labels file *)
    let tab_separated = L.hd (LO.lines_of_file test_labels_fn) in
    let label_strings = S.split_on_char '\t' tab_separated in
    L.map (function
        | "1" -> true
        | "-1" -> false
        | other -> failwith other
      ) label_strings in
  let auc = ROC.auc (L.combine test_labels test_preds) in
  printf "AUC: %.3f\n" auc

let () = main ()
