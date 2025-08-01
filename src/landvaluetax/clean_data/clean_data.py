import pandas as pd
from PyPDF2 import PdfReader
import re
import os
from pathlib import Path
#import pdb # noqa


def clean_eurostat_hpi(df: pd.DataFrame) -> pd.DataFrame:

    # 2) Rename the first column to "country"
    df = df.rename(columns={df.columns[0]: 'country'})

    # 3) Melt wide format (one column per period) into long format
    long = df.melt(
        id_vars=['country'],
        var_name='year_quarter',
        value_name='HPI'
    )

    # 4) Convert HPI values to numeric, drop missing
    long['HPI'] = pd.to_numeric(long['HPI'], errors='coerce')
    long = long.dropna(subset=['HPI'])

    # 5) Split "year_quarter" (e.g. "2020-Q1") into separate year and quarter
    yq = long['year_quarter'].str.split('-Q', expand=True)
    long['year'] = yq[0].astype(int)
    long['quarter'] = yq[1].astype(int)

    # 6) Return only the relevant columns in order
    return long[['year', 'quarter', 'country', 'HPI']]



def clean_prices_estonia(df: pd.DataFrame) -> pd.DataFrame:
    df["municipality"] = (
    df["municipality"]
    .astype(str)            # ensure it’s string dtype
    .str.strip()            # remove leading/trailing spaces
    .str.lower()           # optionally lowercase if casing mismatches
    )

    df["year"] = df["year_quarter"].str.split(" ").str[0].astype(str)
    df["quarter"] = df["year_quarter"].str.split(" ").str[1].astype(str)
    # define a mapping (upper-case keys)
    roman_map = {
        'I':   1,
        'II':  2,
        'III': 3,
        'IV':  4
    }

    # normalize to upper case, map, and (optionally) fill unknowns with NaN
    df['quarter_num'] = df['quarter'].str.upper().map(roman_map)

    return df


def clean_rates_estonia(df: pd.DataFrame) -> pd.DataFrame:

    df["municipality_name"] = (
    df["municipality_name"]
    .astype(str)            # ensure it’s string dtype
    .str.strip()            # remove leading/trailing spaces
    .str.lower()           # optionally lowercase if casing mismatches
    )

    df["year"] = df["year"].astype(str)

    return df

def merge_land_rates_prices(
    land_rates: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:


    prices_clean = clean_prices_estonia(prices)
    rates_clean = clean_rates_estonia(land_rates)

    merged = pd.merge(
        prices_clean,
        rates_clean,
        left_on=["municipality", "year"],
        right_on=["municipality_name", "year"],
        how="inner",    # use "left" or "outer" if you want different join behavior
        validate="many_to_one"  # or "many_to_one" / "one_to_many" as fits your data
    )

    print("Please check the merge between rates and pricdes, still needs some love!") #noqa

    return merged


def merge_eurostat_to_estonia(estonia, eurostat) -> pd.DataFrame:

    eurostat_filter = eurostat[eurostat["country"] == "European Union (EU6-1958, EU9-1973, EU10-1981, EU12-1986, EU15-1995, EU25-2004, EU27-2007, EU28-2013, EU27-2020)"]

    merge_final = pd.merge(
    estonia,
    eurostat_filter,
    left_on=["year", "quarter_num"],
    right_on=["year", "quarter"],
    how="left",
    validate="many_to_one"
    )

    merge_final = merge_final.rename(columns={"country": "country_hpi"})

    return merge_final






def convert_lvt_rate_pdfs() -> pd.DataFrame:
    """
    Convert land value tax rate PDFs to CSV files.

    This function processes PDF files containing land value tax rates for different years,
    extracts the relevant data, and saves it as CSV files.
    """

    # Define years and file paths
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    file_paths = {year: f"/Users/nicolasroever/Dropbox/Promotion/LVT/landvaluetax/src/landvaluetax/data/lvt_rates/maamaksumaarad-{year}.pdf" for year in years}

    final_df = pd.DataFrame()

    # Process each PDF and convert to CSV
    for year in years:
        reader = PdfReader(file_paths[year])

        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        lines = text.splitlines()
        records = []

        if year <= 2023:
            if year == 2020:
                # 2020 includes municipality code
                pattern = r'^\s*(\d+)\s+(.+?)\s{2,}(\d+,\d+|\d+)\s+(\d+,\d+|\d+)\s*$'
                for line in lines:
                    m = re.match(pattern, line)
                    if m:
                        code, name, gen, ag = m.groups()
                        records.append({
                            'municipality_code': code,
                            'municipality_name': name,
                            'general_rate': gen.replace(',', '.'),
                            'agricultural_rate': ag.replace(',', '.'),
                            'year': year
                        })
            else:
                # 2021-2023 no code column
                pattern = r'^\s*(.+?)\s{2,}(\d+,\d+|\d+)\s+(\d+,\d+|\d+)\s*$'
                for line in lines:
                    m = re.match(pattern, line)
                    if m:
                        name, gen, ag = m.groups()
                        records.append({
                            'municipality_name': name,
                            'general_rate': gen.replace(',', '.'),
                            'agricultural_rate': ag.replace(',', '.'),
                            'year': year
                        })
        else:
            # 2024-2025 with three land-use categories
            pattern = r'^\s*(.+?)\s{2,}(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s*$'
            for line in lines:
                m = re.match(pattern, line)
                if m:
                    name, res, ag, oth = m.groups()
                    records.append({
                        'municipality_name': name,
                        'general_rate': res.replace(',', '.'),
                        'agricultural_land_rate': ag.replace(',', '.'),
                        'other_purpose_land_rate': oth.replace(',', '.'),
                        'year': year
                    })

        # Create DataFrame and save CSV
        final_df = pd.concat([final_df, pd.DataFrame(records)], ignore_index=True)

    return final_df



def clean_and_combine_all_price_data() -> pd.DataFrame:
    data_folder = Path("/Users/nicolasroever/Dropbox/Promotion/LVT/landvaluetax/src/landvaluetax/data/raw_data_res_land_quarter")
    i = 0
    all_dfs = []
    for file_path in data_folder.glob("*.xlsx"):
        i += 1
        print(f"Processing file {i}: {file_path.name}") #noqa


        # --- a) Read the transaction_type (first row, first cell) ---
        transaction_type = pd.read_excel(file_path, header=None, nrows=1).iloc[0, 0]

        # --- b) Read the info string (second row, first cell) ---
        info_str = pd.read_excel(file_path, header=None, skiprows=1, nrows=1).iloc[0, 0]

        parts = info_str.split(",", 1)
        county, municipality = parts[0].strip(), parts[1].strip()

        # --- c) Read the data table starting at row 7 (skip 6 rows) ---
        raw = pd.read_excel(file_path, header=None, skiprows=6)
        # drop everything at and after the first blank row
        blank_idx = raw[raw.isna().all(axis=1)].index
        if len(blank_idx):
            raw = raw.loc[: blank_idx[0] - 1]

        # assign your column names
        raw.columns = [
            'year_quarter',
            'house_type',
            "number_trans",
            "average_sqm",
            "total_price",
            "min_price",
            "max_price",
            "min_price_per_sqm",
            "max_price_per_sqm",
            "median_price_per_sqm",
            "average_price_per_sqm",
            "sd_price_per_sqm",
        ]

        # --- d) Add the metadata columns ---
        raw["county"] = county
        raw["municipality"] = municipality


        # collect
        all_dfs.append(raw)

    # 4) Concatenate all into one big DataFrame
    result = pd.concat(all_dfs, ignore_index=True)

    return result
