
open Printf

module A = BatArray
module Fn = Filename
module LO = Line_oriented
module L = BatList
module Utls = OrrandomForest.Utls

module SL = struct
  type t = bool * float (* (label, pred_score) *)
  let get_label (l, _) = l
  let get_score (_, s) = s
end

module ROC = Cpm.MakeROC.Make(SL)

let protect_underscores title =
  BatString.nreplace ~str:title ~sub:"_" ~by:"\\_"

(* comes from RanKers Gnuplot module *)
let roc_curve
    title score_labels_fn roc_curve_fn nb_actives nb_decoys ef_curve_fn =
  let gnuplot_script_fn = Fn.temp_file ~temp_dir:"/tmp" "orrf_" ".gpl" in
  LO.with_out_file gnuplot_script_fn (fun out ->
      fprintf out
        "set title \"|A|:|D|=%d:%d %s\"\n\
         set xtics out nomirror\n\
         set ytics out nomirror\n\
         set size square\n\
         set xrange [0:1]\n\
         set yrange [0:1]\n\
         set xlabel 'ROC: FPR | p_a(m): score_{norm}'\n\
         set ylabel 'TPR'\n\
         set y2label 'p_a(m)'\n\
         set key outside right\n\
         f(x) = x\n\
         g(x) = 1 / (1 + exp(a * x + b))\n\
         fit g(x) '%s' using 1:2 via a, b\n\
         plot '%s' u 1:2 w lines t 'ROC'    , \
              '%s' u 1:2 w lines t '|A|/|D|', \
              ''   u 1:3 w lines t 'A_{%%}' , \
              ''   u 1:4 w lines t 'D_{%%}' , \
              f(x) lc rgb 'black' not, g(x) t 'p_a(m)'\n"
        nb_actives nb_decoys (protect_underscores title)
        score_labels_fn roc_curve_fn ef_curve_fn
    );
  let gnuplot_log = Fn.temp_file ~temp_dir:"/tmp" "gnuplot_" ".log" in
  Utls.run_command (sprintf "(gnuplot -persist %s 2>&1) > %s"
                      gnuplot_script_fn gnuplot_log)
    
let actives_portion_plot_a score_labels =
  (* because thresholds are normalized *)
  let thresholds = L.frange 0.0 `To 1.0 51 in
  let nb_actives = A.count_matching SL.get_label score_labels in
  let nb_decoys = (A.length score_labels) - nb_actives in
  let rev_res =
    L.fold_left (fun acc t ->
        let n =
          A.count_matching
            (fun sl -> SL.get_score sl > t)
            score_labels in
        let card_act =
          A.count_matching
            (fun sl -> (SL.get_label sl) && (SL.get_score sl > t))
            score_labels in
        let card_dec = n - card_act in
        let ef =
          if card_act = 0 || n = 0 then
            0.0 (* there are no more actives above this threshold:
                   the EF falls down to 0.0 (threshold too high) *)
          else (* regular EF formula *)
            (float card_act) /. (float n) in
        let rem_acts = (float card_act) /. (float nb_actives) in
        let rem_decs = (float card_dec) /. (float nb_decoys) in
        (t, ef, rem_acts, rem_decs) :: acc
      ) [] thresholds in
  (nb_actives, nb_decoys, L.rev rev_res)

let performance_plot ?noplot:(noplot = false) title_str for_auc =
  (* save ROC curve *)
  let scores_fn = Fn.temp_file ~temp_dir:"/tmp" "orrf_train_" ".scores" in
  Utls.array_to_file scores_fn
    (fun sl ->
       let score = SL.get_score sl in
       let label = SL.get_label sl in
       sprintf "%f %d" score (Utls.int_of_bool label)
    ) for_auc;
  (* compute ROC curve *)
  let curve_fn = Fn.temp_file ~temp_dir:"/tmp" "orrf_train_" ".roc" in
  let roc_curve_a = ROC.fast_roc_curve_a for_auc in
  Utls.array_to_file curve_fn (fun (x, y) -> sprintf "%f %f" x y)
    roc_curve_a;
  (* plot ROC curve *)
  let ef_curve_fn = Fn.temp_file ~temp_dir:"/tmp" "orrf_train_" ".ef" in
  let nb_acts, nb_decs, ef_curve = actives_portion_plot_a for_auc in
  LO.lines_to_file ef_curve_fn
    (L.map (fun (t, ef, ra, rd) -> sprintf "%f %f %f %f" t ef ra rd) ef_curve);
  (if not noplot then
     roc_curve title_str
       scores_fn curve_fn nb_acts nb_decs ef_curve_fn
  );
