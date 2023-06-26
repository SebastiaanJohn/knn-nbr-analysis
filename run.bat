@echo off

setlocal

set "k=300 600"
set "m=3 7"
set "r_b=1 0.9"
set "r_g=0.6 0.7"
set "a=0.2 0.7 0.9"

set history_path=./data/lastfm_history.csv
set future_path=./data/lastfm_future.csv

for %%k in (%k%) do (
    for %%m in (%m%) do (
        for %%b in (%r_b%) do (
            for %%g in (%r_g%) do (
                for %%a in (%a%) do (
                    python src/eval.py %history_path% %future_path% --k %%k --m %%m --r_b %%b --r_g %%g --alpha %%a > outputs/dataset_lastfm1k_1_250_log_%%k_%%m_%%b_%%g_%%a.txt 2>&1
                )
            )
        )
    )
)

endlocal