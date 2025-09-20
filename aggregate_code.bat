@echo off
echo Aggregating code files for review...
echo. > aggregated_code_for_review.txt

echo --- convert_spelling.py --- >> aggregated_code_for_review.txt
type convert_spelling.py >> aggregated_code_for_review.txt
echo. >> aggregated_code_for_review.txt

echo --- model.py --- >> aggregated_code_for_review.txt
type model.py >> aggregated_code_for_review.txt
echo. >> aggregated_code_for_review.txt

echo --- train_llm.py --- >> aggregated_code_for_review.txt
type train_llm.py >> aggregated_code_for_review.txt
echo. >> aggregated_code_for_review.txt

echo --- inference.py --- >> aggregated_code_for_review.txt
type inference.py >> aggregated_code_for_review.txt
echo. >> aggregated_code_for_review.txt

echo Aggregation complete. Output saved to aggregated_code_for_review.txt
pause