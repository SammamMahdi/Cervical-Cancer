# -*- coding: utf-8 -*-
import io, docx, openpyxl
KW=("fus","p_final","p_risk","p_cell","tier","weighted","decision","ensemble","concat","attention","transformer","multimodal","risk","rf","logistic","alpha","α","both input","modality")
def hit(s): 
    s=s.lower(); return any(k in s for k in KW)
print("######## DOCX: Cervical Cancer Paper Pipeline ########")
d=docx.Document(r"legacy/Cervical Cancer Paper Pipeline.docx")
for p in d.paragraphs:
    t=p.text.strip()
    if t and hit(t): print(" P:", t)
for ti,tb in enumerate(d.tables):
    for row in tb.rows:
        cells=[c.text.strip().replace("\n"," ") for c in row.cells]
        line=" | ".join(cells)
        if hit(line): print(f" T{ti}:", line)
print("\n######## XLSX: Cervical Cancer Research (fusion-relevant cells) ########")
wb=openpyxl.load_workbook(r"legacy/Cervical Cancer Research.xlsx", data_only=True)
for sn in wb.sheetnames:
    ws=wb[sn]; printed=False
    for row in ws.iter_rows(values_only=True):
        for cell in row:
            if isinstance(cell,str) and hit(cell) and ("fus" in cell.lower() or "p_final" in cell.lower() or "tier" in cell.lower() or "multimodal" in cell.lower() or "concat" in cell.lower()):
                if not printed: print(f"\n--- sheet: {sn} ---"); printed=True
                seg=cell.strip().replace("\n"," ")
                print("  *", (seg[:400]+("..." if len(seg)>400 else "")))
