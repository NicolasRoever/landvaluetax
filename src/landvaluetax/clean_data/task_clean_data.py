import pytask
import re
import pandas as pd
#from PyPDF2 import PdfReader
import pandas as pd
from pathlib import Path
#import pdb # noqa

from src.landvaluetax.config import BLD, SRC


from src.landvaluetax.clean_data.clean_data import convert_lvt_rate_pdfs, clean_eurostat_hpi, merge_land_rates_prices, merge_eurostat_to_estonia, clean_and_combine_all_price_data



def task_create_quarterly_prices_res_land(
    produces: Path = BLD / "data" / "quarterly_prices_res_land.csv"):

    clean_and_combine_all_price_data().to_csv(produces, index=False)





def task_merge_estonia_eurostat(
    estonia = BLD / "data" / "merged_rates_prices.csv",
    eurostat_filter = BLD / "data" / "eurostat_hpi.csv",
    produces: Path = BLD / "data" / "merged_estonia_eurostat.csv"
) -> None:

    estonia_df = pd.read_csv(estonia)
    eurostat_df = pd.read_csv(eurostat_filter)

    merged_df = merge_eurostat_to_estonia(estonia_df, eurostat_df)

    merged_df.to_csv(produces, index=False)



def task_clean_land_rates_prices(
    depends_on= [BLD / "data" / "quarterly_prices_res_land.csv", BLD / "data" / "lvt_rates.csv"],
    produces = BLD / "data" / "merged_rates_prices.csv"
) -> None:

    prices = pd.read_csv(depends_on[0])
    rates = pd.read_csv(depends_on[1])

    merged = merge_land_rates_prices(
        land_rates=rates,
        prices=prices
    )

    merged.to_csv(produces, index=False)

def task_clean_eurostat_hpi(
    produces: Path = BLD / "data" / "eurostat_hpi.csv",
    depends_on: Path = SRC / "data" / "eurostat_hpi" / "eurostat_hpi.xlsx"):

    raw_data = pd.read_excel(depends_on, sheet_name="Sheet 1", skiprows=8)

    cleaned_data = clean_eurostat_hpi(raw_data)

    cleaned_data.to_csv(produces, index=False)


def task_convert_lvt_rate_pdfs(
    produces: Path = BLD / "data" / "lvt_rates.csv"
) -> None:
    """
    Task to convert land value tax rate PDFs to CSV files.

    This task processes PDF files containing land value tax rates for different years,
    extracts the relevant data, and saves it as CSV files.
    """


    df = convert_lvt_rate_pdfs()

    df.to_csv(produces)
