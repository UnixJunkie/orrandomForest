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
  let sparse_data_fn = "data/train_data.csr" in
  let labels_fn = "data/train_labels.txt" in
  let preds =
    let params = Rf.(default_params 1831 Class) in
    let sparsity = Rf.Dense in
    let model =
      Rf.train
        ~debug:true
        sparsity
        params
        data_fn
        labels_fn in
    Rf.read_predictions
      (Rf.predict ~debug:true sparsity model data_fn) in
  let sparse_preds =
    let params = Rf.(default_params 1831 Class) in
    let sparsity = Rf.Sparse 1831 in
    let model =
      Rf.train
        ~debug:true
        sparsity
        params
        sparse_data_fn
        labels_fn in
    Rf.read_predictions
      (Rf.predict ~debug:true sparsity model sparse_data_fn) in
  assert(List.length preds = 88);
  assert(List.length sparse_preds = 88);
  let labels =
    let labels_line = Utls.with_in_file labels_fn input_line in
    let label_strings = BatString.split_on_char '\t' labels_line in
    L.map (function
        | "1" -> true
        | "-1" -> false
        | other -> failwith other
      ) label_strings in
  let auc = ROC.auc (List.combine labels preds) in
  let sparse_auc = ROC.auc (List.combine labels sparse_preds) in
  printf "AUC: %.3f\n" auc;
  printf "sparse AUC: %.3f\n" sparse_auc

let () = main ()
