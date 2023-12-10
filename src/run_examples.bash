echo GOLEM-EV

python3 main.py --method GOLEM \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo GOLEM-NV

python3 main.py  --method GOLEM \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo GOLEM-NV with Initializetion

python3 main.py  --method GOLEM \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

python3 main.py  --method GOLEM \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --non_equal_variances \
                     --init \
                     --lambda_1 2e-3 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo GOLEM-EV with Using pyTorch implementation

python3 main.py  --method GOLEM_TORCH \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo GOLEM-NV with Using pyTorch implementation

python3 main.py  --method GOLEM_TORCH \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --non_equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo GOLEM-NV with Early stop

python3 main.py  --method GOLEM_TORCH \
                     --early_stop_delta 1e-4 \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_nv \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

echo DAGMA

python3 main.py  --method DAGMA \
                     --seed 1 \
                     --d 10 \
                     --lambda_1 2e-2 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --checkpoint_iter 1000 \
                     --loss l2

echo NOTEARS

python3 main.py  --method NOTEARS \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --loss l2
