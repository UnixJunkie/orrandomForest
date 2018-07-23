open Printf

module Rf = OrrandomForest.Rf
module L = BatList
module Utls = OrrandomForest.Utls

module Score_label = struct
  type t = bool * float (* (label, pred_score) *)
  let get_label (l, _) = l
  let get_score (_, s) = s
end

module ROC = Cpm.MakeROC.Make(Score_label)

let main () =
  Log.set_log_level Log.DEBUG;
  Log.color_on ();
  let data_fn = "data/train_data.txt" in
  let labels_fn = "data/train_labels.txt" in
  let preds =
    let params = Rf.(default_params 1831 Class) in
    let model =
      Rf.train
        ~debug:true
        params
        data_fn
        labels_fn in
    Rf.read_predictions
      (Rf.predict ~debug:true model data_fn) in
  assert(List.length preds = 88);
  let labels =
    let labels_line = Utls.with_in_file labels_fn input_line in
    let label_strings = BatString.split_on_char '\t' labels_line in
    L.map (function
        | "1" -> true
        | "-1" -> false
        | other -> failwith other
      ) label_strings in
  let auc = ROC.auc (List.combine labels preds) in
  printf "AUC: %.3f\n" auc

let () = main ()
