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

let run_command verbose (cmd: string): unit =
  if verbose then
    Log.info "run_command: %s" cmd
  ;
  match Unix.system cmd with
  | Unix.WSIGNALED _ -> (Log.fatal "run_command: signaled: %s" cmd; exit 1)
  | Unix.WSTOPPED _ -> (Log.fatal "run_command: stopped: %s" cmd; exit 1)
  | Unix.WEXITED i when i <> 0 ->
    (Log.fatal "run_command: exit %d: %s" i cmd; exit 1)
  | Unix.WEXITED _ (* i = 0 then *) -> ()

let train_classifier verbose trees features data_fn labels_fn =
  Rf.(train ~debug:verbose
        Classification
        Rf.Dense
        (* parameters *)
        { ntree = trees;
          mtry = features;
          importance = true }
        data_fn
        labels_fn)

let test_classifier verbose trained_model data_fn maybe_preds_out_fn =
  Rf.(read_predictions
        (predict ~debug:verbose Classification Dense trained_model
           data_fn maybe_preds_out_fn))

let split_label_features verbose data_csv_fn =
  let tmp_labels_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_labels_" ".csv" in
  let tmp_features_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_features_" ".csv" in
  let cmd1 = sprintf "cut -d' ' -f1 %s > %s" data_csv_fn tmp_labels_fn in
  let cmd2 = sprintf "cut -d' ' -f2- %s | tr ' ' '\t' > %s" data_csv_fn tmp_features_fn in
  run_command verbose cmd1;
  run_command verbose cmd2;
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

let train_test verbose trees features index2feature_name train_lines test_lines =
  let tmp_train_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_train_" ".csv" in
  let tmp_test_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_test_" ".csv" in
  LO.lines_to_file tmp_train_fn train_lines;
  LO.lines_to_file tmp_test_fn test_lines;
  let train_features_fn, train_labels_fn = split_label_features verbose tmp_train_fn in
  let model = train_classifier verbose trees features train_features_fn train_labels_fn None in
  L.iter Sys.remove [tmp_train_fn; train_features_fn; train_labels_fn];
  let feat_importance, imp_fn = Rf.read_predictions (Rf.get_features_importance model) in
  Sys.remove imp_fn;
  assert(L.length feat_importance = A.length index2feature_name);
  L.iteri (fun i imp ->
      Log.info "imp(%s): %.2f" index2feature_name.(i) imp
    ) feat_importance;
  let test_features_fn, test_labels_fn = split_label_features verbose tmp_test_fn in
  let test_preds, preds_fn = test_classifier verbose model test_features_fn None in
  Sys.remove preds_fn;
  let test_labels =
    (* all labels are on a single line in the labels file *)
    let tab_separated = L.hd (LO.lines_of_file test_labels_fn) in
    let label_strings = S.split_on_char '\t' tab_separated in
    L.map (function
        | "1" -> true
        | "-1" -> false
        | other -> failwith other
      ) label_strings in
  L.iter Sys.remove [tmp_test_fn; test_features_fn; test_labels_fn];
  L.combine test_labels test_preds

let only_train verbose trees features index2feature_name train_lines maybe_model_out_fn =
  let tmp_train_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_train_" ".csv" in
  LO.lines_to_file tmp_train_fn train_lines;
  let train_features_fn, train_labels_fn = split_label_features verbose tmp_train_fn in
  let model = train_classifier verbose trees features train_features_fn train_labels_fn maybe_model_out_fn in
  L.iter Sys.remove [tmp_train_fn; train_features_fn; train_labels_fn];
  let feat_importance, imp_fn = Rf.read_predictions (Rf.get_features_importance model) in
  Sys.remove imp_fn;
  assert(L.length feat_importance = A.length index2feature_name);
  L.iteri (fun i imp ->
      Log.info "imp(%s): %.2f" index2feature_name.(i) imp
    ) feat_importance;
  model

let only_predict verbose trained_model test_lines maybe_preds_out_fn =
  let tmp_test_fn = Fn.temp_file ~temp_dir:"/tmp" "classif_test_" ".csv" in
  LO.lines_to_file tmp_test_fn test_lines;
  let test_features_fn, test_labels_fn = split_label_features verbose tmp_test_fn in
  let _test_preds, preds_fn = test_classifier verbose trained_model test_features_fn maybe_preds_out_fn in
  (match maybe_preds_out_fn with
   | None -> Sys.remove preds_fn
   | Some _ -> ()
  );
  L.iter Sys.remove [tmp_test_fn; test_features_fn; test_labels_fn]

type model_file_mode = Save_to of string
                     | Load_from of string
                     | Ignore

(* [--no-plot]: turn OFF ROC curve\n *)

let main () =
  let argc, args = CLI.init () in
  Log.(set_log_level INFO);
  Log.color_on ();
  Log.(set_prefix_builder short_prefix_builder);
  let train_portion_def = 0.8 in
  let nb_trees_def = 100 in
  if argc = 1 || CLI.get_set_bool ["-h";"--help"] args then
    begin
      eprintf "usage:\n\
               %s  \
               -i <train.csv>: training set\n  \
               [-p <float>]: proportion of the (randomized) dataset\n  \
               used to train (default=%.2f)\n  \
               [--seed <int>: fix random seed]\n  \
               [-n <int>]: num_trees=|RF|; default=%d\n  \
               [--mtry <float>]: proportion of randomly selected features\n  \
               to use at each split (default=(sqrt(|feats|))/|feats|)\n  \
               [--scan-mtry]: scan for best mtry in [0.001,0.002,0.005,...,1.0]\n  \
               (incompatible with --mtry)\n  \
               [--mtry-range <string>]: mtrys to test e.g. \"0.001,0.002,0.005\"\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-np <int>]: max number of processes for --NxCV (default=1)\n  \
               [-s <filename>]: save model to file after training\n  \
               [-l <filename>]: load trained model from file\n  \
               [-o <filename>]: save predictions to file\n  \
               [-v]: verbose/debug mode\n"
        Sys.argv.(0) train_portion_def nb_trees_def;
      exit 1
    end;
  let input_fn = CLI.get_string ["-i"] args in
  let rng = match CLI.get_int_opt ["--seed"] args with
    | None -> Random.State.make_self_init ()
    | Some seed -> Random.State.make [|seed|] in
  let nb_trees = CLI.get_int_def ["-n"] args nb_trees_def in
  let verbose = CLI.get_set_bool ["-v"] args in
  let maybe_mtry = CLI.get_float_opt ["--mtry"] args in
  let scan_mtry = CLI.get_set_bool ["--scan-mtry"] args in
  let mtry_range = CLI.get_string_opt ["--mtry-range"] args in
  let cv_folds = CLI.get_int_def ["--NxCV"] args 1 in
  let nprocs = CLI.get_int_def ["-np"] args 1 in
  let maybe_preds_out_fn = CLI.get_string_opt ["-o"] args in
  let p, mode =
    begin match CLI.get_string_opt ["-l"] args with
      | Some fn ->
        (* no training portion *)
        let () = Log.info "p forced to 0.0" in
        (0.0, Load_from fn)
      | None ->
        begin match CLI.get_string_opt ["-s"] args with
          | Some fn ->
            (* training on whole dataset *)
            let () = Log.info "p forced to 1.0" in
            (1.0, Save_to fn)
          | None ->
            let p = CLI.get_float_def ["-p"] args train_portion_def in
            (p, Ignore)
        end
    end in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let header, all_lines =
    let lines = LO.lines_of_file input_fn in
    match lines with
    | header' :: data_lines ->
      assert(S.starts_with header' "#");
      (S.lchop header',
       if p > 0.0 then
         (* if some model training is to be done: shuffle training set *)
         L.shuffle ~state:rng data_lines
       else
         (* when predicting in production: DO NOT shuffle lines *)
         data_lines
      )
    | _ -> failwith ("not enough lines in: " ^ input_fn) in
  Log.info "header: %s" header;
  let nb_features = S.count_char header ' ' in
  let nb_feats = float nb_features in
  Log.info "|features|=%d" nb_features;
  let mtrys = match (maybe_mtry, scan_mtry, mtry_range) with
    | (Some mtry', false, None) -> [mtry'] (* single mtry value *)
    | (None, true, None) -> (* exponential scan *)
      (* high values first, for better parallelization if not enough cores
         (they'll take longer to complete) *)
      L.rev [0.001; 0.002; 0.005;
             0.01 ; 0.02 ; 0.05 ;
             0.1  ; 0.2  ; 0.5  ; 1.0]
    | (None, false, Some range_str) ->
      L.map (fun x_str -> float_of_string x_str)
        (S.split_on_char ',' range_str)
    | (None, false, None) ->
      [(sqrt nb_feats) /. nb_feats]
    | _ -> failwith "Model.main: only one of {--mtry|--scan-mtry|--mtry-range}" in
  let index2feature_name = A.of_list (L.tl (S.split_on_char ' ' header)) in
  assert(A.length index2feature_name = nb_features);
  let nprocs1 =
    (* choose at which level there will be parallelization *)
    if L.length mtrys >= cv_folds then nprocs
    else 1 in
  Parany.Parmap.pariter nprocs1 (fun mtry ->
      (* apply mtry param to nb_features *)
      let features =
        let x = BatFloat.round_to_int (mtry *. nb_feats) in
        (* constrain nb_features in [1, nb_features] *)
        min nb_features (max x 1) in
      Log.info "using %d/%d features" features nb_features;
      let label_scores =
        if cv_folds <= 1 then
          let n = L.length all_lines in
          let train_n, test_n =
            let x = BatFloat.round_to_int (p *. (float_of_int n)) in
            (x, n - x) in
          Log.info "train/test: %d/%d" train_n test_n;
          let train_lines, test_lines = L.takedrop train_n all_lines in
          match mode with
          | Ignore ->
            train_test
              verbose nb_trees features index2feature_name train_lines test_lines
          | Save_to model_out_fn ->
            let _trained_model =
              only_train
                verbose nb_trees features index2feature_name train_lines (Some model_out_fn) in
            [] (* no predictions *)
          | Load_from model_fn ->
            let () = only_predict verbose (Ok model_fn) test_lines maybe_preds_out_fn in
            [] (* production mode: we don't know the true labels so we cannot combine
                  them with the predictions *)
        else (* cv_folds > 1 *)
          let folds = Cpm.Utls.cv_folds cv_folds all_lines in
          let for_auc =
            let nprocs2 =
              if cv_folds > L.length mtrys then nprocs
              else 1 (* do not parallelize inside of a parallel loop *) in
            Parany.Parmap.parmap nprocs2 (fun (train_lines, test_lines) ->
                train_test verbose nb_trees features index2feature_name train_lines test_lines
              ) folds in
          L.concat for_auc in
      match label_scores with
      | [] -> () (* only train or only predict: performance cannot be estimated *)
      | _ ->
        let auc = ROC.auc label_scores in
        Log.info "|trees|=%d mtry=%g feats=%d AUC: %.3f\n" nb_trees mtry features auc
    ) mtrys

let () = main ()
