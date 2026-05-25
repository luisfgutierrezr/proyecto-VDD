from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
HURTOS_PATH = BASE / "hurtos_bogota_2025.csv"
LLAMADAS_JSON = BASE / "Llamadas123C4.json"
LLAMADAS_CSV = BASE / "Llamadas123C4.csv"
UPZ_JSON = BASE / "DAIUPZ.json"
UPZ_GEOJSON = BASE / "DAIUPZ.geojson"

DIAS_ES = [
    "LUNES",
    "MARTES",
    "MIÉRCOLES",
    "JUEVES",
    "VIERNES",
    "SÁBADO",
    "DOMINGO",
]

LAT_MIN, LAT_MAX = 4.35, 4.95
LON_MIN, LON_MAX = -74.25, -73.95

UPZ_SENTINELS = {
    "UPZ999",
    "UPZ990",
    "UPZ991",
    "UPZ993",
    "UPZ994",
    "UPZ995",
    "UPZ996",
    "-",
}


def parse_fecha_dmY(value: str) -> pd.Timestamp | pd.NaT:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    parts = str(value).strip().split("/")
    if len(parts) != 3:
        return pd.NaT
    d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
    if y < 100:
        y += 2000 if y < 70 else 1900
    return pd.Timestamp(year=y, month=m, day=d)


def parse_fecha_dmyy(value: str) -> pd.Timestamp | pd.NaT:
    return parse_fecha_dmY(value)


def clean_hurtos() -> dict:
    raw = pd.read_csv(HURTOS_PATH, sep=";", encoding="latin-1")
    stats = {"filas_origen": len(raw)}

    raw.columns = [
        "armas_medios",
        "departamento",
        "municipio",
        "fecha_hecho_raw",
        "genero",
        "agrupa_edad",
        "codigo_dane",
        "cantidad",
    ]

    for col in ("armas_medios", "genero", "agrupa_edad", "departamento", "municipio"):
        raw[col] = raw[col].astype(str).str.strip()
        raw[col] = raw[col].replace({"": "SIN_DATO", "nan": "SIN_DATO"})

    raw["cantidad"] = pd.to_numeric(raw["cantidad"], errors="coerce").fillna(1).astype(int)
    raw["codigo_dane"] = raw["codigo_dane"].astype(str).str.strip()

    raw["fecha_hecho"] = raw["fecha_hecho_raw"].apply(parse_fecha_dmY)
    invalid_dates = raw["fecha_hecho"].isna().sum()
    raw = raw.dropna(subset=["fecha_hecho"])
    raw["fecha_hecho"] = raw["fecha_hecho"].dt.strftime("%Y-%m-%d")
    raw["anio"] = raw["fecha_hecho"].str.slice(0, 4).astype(int)
    raw["mes"] = raw["fecha_hecho"].str.slice(5, 7).astype(int)
    raw["dia_semana"] = pd.to_datetime(raw["fecha_hecho"]).dt.dayofweek + 1
    raw["nombre_dia"] = pd.to_datetime(raw["fecha_hecho"]).dt.dayofweek.map(
        lambda i: DIAS_ES[i]
    )
    raw["nivel_geografico"] = "municipio"

    agg = (
        raw.groupby(
            [
                "fecha_hecho",
                "anio",
                "mes",
                "dia_semana",
                "nombre_dia",
                "genero",
                "agrupa_edad",
                "armas_medios",
                "departamento",
                "municipio",
                "codigo_dane",
                "nivel_geografico",
            ],
            as_index=False,
        )["cantidad"]
        .sum()
        .sort_values(["fecha_hecho", "genero", "agrupa_edad", "armas_medios"])
    )

    agg.to_csv(HURTOS_PATH, sep=";", index=False, encoding="utf-8")
    stats.update(
        {
            "filas_limpias": len(agg),
            "fechas_invalidas_descartadas": int(invalid_dates),
            "total_cantidad": int(agg["cantidad"].sum()),
        }
    )
    return stats


def _parse_coord(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace(",", ".")
    if s in ("", "0", "0.0"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def clean_llamadas(valid_upz: set[str]) -> dict:
    with open(LLAMADAS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    fields = [x["id"] for x in data["fields"]]
    df = pd.DataFrame(data["records"], columns=fields)
    stats = {"filas_origen": len(df)}

    rename = {
        "FECHA": "fecha_raw",
        "SEMANA_ISO": "semana_iso",
        "NUM_DIA_SEMANA": "num_dia_semana",
        "NOMBRE_DIA": "nombre_dia",
        "ANIO": "anio",
        "MES": "mes",
        "RANGO_HORA": "rango_hora",
        "RANGO_DEL_DIA": "rango_del_dia",
        "ID_INCIDENTE": "id_incidente",
        "TIPO_INCIDENTE": "tipo_incidente",
        "DESC_TIPO_INCIDENTE": "desc_tipo_incidente",
        "CIRCUNSTANCIA_INCIDENTE": "circunstancia_incidente",
        "COD_LOCALIDAD": "cod_localidad",
        "LOCALIDAD": "localidad",
        "COD_UPZ": "cod_upz_raw",
        "UPZ": "nombre_upz",
        "COD_SEC_CATAST": "cod_sec_catastral",
        "SEC_CATASTRAL": "sec_catastral",
        "COD_BARRIO": "cod_barrio",
        "BARRIO": "barrio",
        "LATITUD": "latitud_raw",
        "LONGITUD": "longitud_raw",
        "DIRECCION": "direccion",
        "COD_AGENCIA": "cod_agencia",
        "AGENCIA": "agencia",
    }
    keep = [c for c in rename if c in df.columns]
    df = df[keep].rename(columns={k: rename[k] for k in keep})

    df["nombre_dia"] = df["nombre_dia"].astype(str).str.strip()
    df["localidad"] = df["localidad"].astype(str).str.strip()
    df["nombre_upz"] = df["nombre_upz"].astype(str).str.strip()
    df["cod_upz"] = df["cod_upz_raw"].astype(str).str.strip().str.upper()

    df["fecha"] = df["fecha_raw"].apply(parse_fecha_dmyy)
    df = df.dropna(subset=["fecha"])
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")

    df["latitud"] = df["latitud_raw"].apply(_parse_coord)
    df["longitud"] = df["longitud_raw"].apply(_parse_coord)

    df["sin_localizacion"] = (
        (df["cod_upz"].isin(UPZ_SENTINELS))
        | (df["latitud"] == 0)
        | (df["longitud"] == 0)
    ).astype(int)

    df["coord_valida"] = (
        (df["sin_localizacion"] == 0)
        & df["latitud"].between(LAT_MIN, LAT_MAX)
        & df["longitud"].between(LON_MIN, LON_MAX)
    ).astype(int)

    df["upz_valida"] = (
        (df["sin_localizacion"] == 0) & df["cod_upz"].isin(valid_upz)
    ).astype(int)

    before_dedup = len(df)
    df = df.sort_values(
        ["id_incidente", "coord_valida", "upz_valida"],
        ascending=[True, False, False],
    )
    df = df.drop_duplicates(subset=["id_incidente"], keep="first")
    stats["duplicados_id_incidente_eliminados"] = before_dedup - len(df)

    out_cols = [
        "id_incidente",
        "fecha",
        "anio",
        "mes",
        "semana_iso",
        "num_dia_semana",
        "nombre_dia",
        "rango_hora",
        "rango_del_dia",
        "tipo_incidente",
        "desc_tipo_incidente",
        "circunstancia_incidente",
        "cod_localidad",
        "localidad",
        "cod_upz",
        "nombre_upz",
        "cod_barrio",
        "barrio",
        "latitud",
        "longitud",
        "direccion",
        "cod_agencia",
        "agencia",
        "sin_localizacion",
        "coord_valida",
        "upz_valida",
    ]
    df[out_cols].to_csv(LLAMADAS_CSV, sep=";", index=False, encoding="utf-8")
    LLAMADAS_JSON.unlink()

    stats.update(
        {
            "filas_limpias": len(df),
            "sin_localizacion": int(df["sin_localizacion"].sum()),
            "coord_valida": int(df["coord_valida"].sum()),
            "upz_valida": int(df["upz_valida"].sum()),
            "anios": df["anio"].astype(str).value_counts().sort_index().to_dict(),
        }
    )
    return stats


def clean_upz() -> tuple[set[str], dict]:
    gdf = gpd.read_file(UPZ_JSON)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf_proj = gdf.to_crs("EPSG:9377")
    area_km2 = gdf_proj.geometry.area / 1_000_000

    out = gpd.GeoDataFrame(
        {
            "objectid": gdf["OBJECTID"].astype(int),
            "cod_upz": gdf["CMIUUPLA"].astype(str).str.strip().str.upper(),
            "nombre_upz": gdf["CMNOMUPLA"].astype(str).str.strip(),
            "poblacion_proxy": pd.to_numeric(gdf["CMHPTOTAL"], errors="coerce"),
            "area_km2": area_km2.round(4),
        },
        geometry=gdf.geometry,
        crs="EPSG:4326",
    )

    valid_upz = set(out["cod_upz"].unique())
    out.to_file(UPZ_GEOJSON, driver="GeoJSON")
    UPZ_JSON.unlink()

    stats = {
        "poligonos": len(out),
        "cod_upz_unicos": len(valid_upz),
    }
    return valid_upz, stats


def main() -> None:
    print("=== Limpieza datasets VDD ===\n")
    results = {}

    print("1/3 UPZ (DAIUPZ.json -> DAIUPZ.geojson)...")
    valid_upz, results["upz"] = clean_upz()
    print(f"   {results['upz']}\n")

    print("2/3 Hurtos (hurtos_bogota_2025.csv)...")
    results["hurtos"] = clean_hurtos()
    print(f"   {results['hurtos']}\n")

    print("3/3 Llamadas (Llamadas123C4.json -> Llamadas123C4.csv)...")
    results["llamadas"] = clean_llamadas(valid_upz)
    print(f"   {results['llamadas']}\n")

    report_path = BASE / "_clean_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now().isoformat(), **results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Reporte: {report_path}")
    print("Listo.")


if __name__ == "__main__":
    main()
    sys.exit(0)
