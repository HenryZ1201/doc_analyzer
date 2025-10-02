2025-10-01
- added function to render latex format table to matplotlib plottable tables, and count the ncols and nrows
- adjusted StructEqTable 'max_new_tokens' to 4096, to support longger tables

2025-09-30
- Used yolo-based model for layout recognition
- Use table parsing model StructEqTable to extract informations from table ad convert to latex formated .txt files.

** To Do **
- replace StructEqTable with a beter table recognition model? like Unitable model?