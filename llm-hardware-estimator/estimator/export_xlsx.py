import html
import re
import zipfile
from pathlib import Path


def _col_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _safe_sheet_name(name):
    cleaned = re.sub(r"[][\\/*?:]", "_", str(name))[:31]
    return cleaned or "Sheet"


def _sheet_xml(rows):
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    table = [dict((key, key) for key in keys)] + list(rows)
    xml_rows = []
    for row_idx, row in enumerate(table, 1):
        cells = []
        for col_idx, key in enumerate(keys, 1):
            value = row.get(key, "")
            ref = "%s%d" % (_col_name(col_idx), row_idx)
            if isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf")):
                cells.append('<c r="%s"><v>%s</v></c>' % (ref, value))
            else:
                text = html.escape("" if value is None else str(value))
                cells.append('<c r="%s" t="inlineStr"><is><t>%s</t></is></c>' % (ref, text))
        xml_rows.append('<row r="%d">%s</row>' % (row_idx, "".join(cells)))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>%s</sheetData></worksheet>' % "".join(xml_rows)
    )


def write_xlsx(path, sheets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [(_safe_sheet_name(name), list(rows)) for name, rows in sheets.items()]
    if not normalized:
        normalized = [("Summary", [])]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                '<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                % idx
                for idx in range(1, len(normalized) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        sheets_xml = "".join(
            '<sheet name="%s" sheetId="%d" r:id="rId%d"/>'
            % (html.escape(name), idx, idx)
            for idx, (name, _) in enumerate(normalized, 1)
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>%s</sheets></workbook>" % sheets_xml,
        )
        rels_xml = "".join(
            '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>'
            % (idx, idx)
            for idx in range(1, len(normalized) + 1)
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s</Relationships>'
            % rels_xml,
        )
        for idx, (_, rows) in enumerate(normalized, 1):
            zf.writestr("xl/worksheets/sheet%d.xml" % idx, _sheet_xml(rows))
