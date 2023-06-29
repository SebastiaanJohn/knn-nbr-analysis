@echo off

setlocal

set "k=300 600"
set "m=3 7 11"
set "r_b=0.1 0.5 0.9"
set "r_g=0.1 0.5 0.9"
set "a=0.3 0.5 0.7 0.9"

set history_path=./data/lastfm-1b_history.csv
set future_path=./data/lastfm-1b_future.csv

for %%k in (%k%) do (
    for %%m in (%m%) do (
        for %%b in (%r_b%) do (
            for %%g in (%r_g%) do (
                for %%a in (%a%) do (
                    python src/eval.py %history_path% %future_path% --k %%k --m %%m --r_b %%b --r_g %%g --alpha %%a > outputs/dataset_lastfm1b_log_%%k_%%m_%%b_%%g_%%a.txt 2>&1
                )
            )
        )
    )
)

endlocal