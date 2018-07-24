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

let test_classification () =
  let data_fn = "data/train_data.txt" in
  let sparse_data_fn = "data/train_data.csr" in
  let labels_fn = "data/train_labels.txt" in
  let preds =
    let params = Rf.(default_params 1831 Class) in
    let sparsity = Rf.Dense in
    let model =
      Rf.(train
            ~debug:true
            Class
            sparsity
            params
            data_fn
            labels_fn) in
    Rf.(read_predictions
          (predict ~debug:true Class sparsity model data_fn)) in
  let sparse_preds =
    let params = Rf.(default_params 1831 Class) in
    let sparsity = Rf.Sparse 1831 in
    let model =
      Rf.(train
            ~debug:true
            Class
            sparsity
            params
            sparse_data_fn
            labels_fn) in
    Rf.(read_predictions
          (predict ~debug:true Class sparsity model sparse_data_fn)) in
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

let rmse l1 l2 =
  let l = L.combine l1 l2 in
  let n = float (L.length l1) in
  let sum_diff2 =
    L.fold_left (fun acc (x, y) ->
        let diff = x -. y in
        acc +. (diff *. diff)
      ) 0.0 l in
  sqrt (sum_diff2 /. n)

let test_regression () =
  let train_features_fn = "data/Boston_train_features.csv" in
  let train_values_fn = "data/Boston_train_values.csv" in
  let test_features_fn = "data/Boston_test_features.csv" in
  let test_features_sparse_fn = "data/Boston_test_features.csr" in
  let params = Rf.(default_params 13 Regre) in
  let preds =
    let sparsity = Rf.Dense in
    let model =
      Rf.(train
            ~debug:true
            Regre
            sparsity
            params
            train_features_fn
            train_values_fn) in
    Rf.(read_predictions
          (predict ~debug:true Regre sparsity model test_features_fn)) in
  let sparse_preds =
    let train_features_sparse_fn = "data/Boston_train_features.csr" in
    let sparsity = Rf.Sparse 13 in
    let model =
      Rf.(train
            ~debug:true
            Regre
            sparsity
            params
            train_features_sparse_fn
            train_values_fn) in
    Rf.(read_predictions
          (predict ~debug:true Regre sparsity model test_features_sparse_fn)) in
  let actual = Utls.float_list_of_file "data/Boston_test_values.csv" in
  assert(List.length actual = List.length preds);
  assert(List.length preds = 50);
  assert(List.length sparse_preds = 50);
  let err = rmse preds actual in
  let sparse_err = rmse sparse_preds actual in
  printf "test set RMSE: %.3f\n" err;
  printf "sparse test set RMSE: %.3f\n" sparse_err
  (* FBR: plot actual versus predicted and fit *)

let main () =
  Log.set_log_level Log.DEBUG;
  Log.color_on ();
  test_regression ();
  test_classification ()

let () = main ()
