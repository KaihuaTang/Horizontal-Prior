python paired_bootstrap.py \
    --csv_a exp/test_idcue_constraint/per_sample_rotate_0_90.csv \
    --csv_b exp/test_baseline/per_sample_rotate_0_90.csv \
    --name_a "IDCue-Constraint" --name_b "Baseline" \
    --metrics d1 abs_rel rmse \
    --B 1000 --seed 0 \
    --save_md report_tipping.md