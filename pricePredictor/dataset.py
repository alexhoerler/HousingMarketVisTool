import csv
from datetime import datetime
import torch
from torch.utils.data import Dataset


class PricePredictorDataset(Dataset):
    def __init__(self, data_file):
        self.housing_types = ["Single Family Residential", "Townhouse",
                              "Condo/Co-op", "All Residential", "Multi-Family (2-4 Unit)"]
        self.data_points = dict()
        with open(data_file, "r") as file:
            csv_reader = csv.reader(file)

            for header in csv_reader:
                break
            for row in csv_reader:
                start_date = datetime.strptime(row[0], "%Y-%m-%d")
                start_date = datetime.strptime(row[1], "%Y-%m-%d")
                state = row[2]
                median_ppsf = float(row[3]) if row[3] != "" else 0.0
                median_list_ppsf = float(row[4]) if row[4] != "" else 0.0
                median_sale_price = float(row[5]) if row[5] != "" else 0.0
                homes_sold = int(row[6]) if row[6] != "" else 0.0
                new_listings = float(row[7]) if row[7] != "" else 0.0
                median_dom = float(row[8]) if row[8] != "" else 0.0
                property_type = row[9]

                state_dict = self.data_points.get(state, dict())
                state_year_dict = state_dict.get(start_date.year, dict())

                avg_median_ppsf = state_year_dict.get("avg_median_ppsf", [])
                avg_median_ppsf.append(median_ppsf)
                state_year_dict["avg_median_ppsf"] = avg_median_ppsf

                avg_median_list_ppsf = state_year_dict.get(
                    "avg_median_list_ppsf", [])
                avg_median_list_ppsf.append(median_list_ppsf)
                state_year_dict["avg_median_list_ppsf"] = avg_median_list_ppsf

                avg_median_sale_price = state_year_dict.get(
                    "avg_median_sale_price", [])
                avg_median_sale_price.append(median_sale_price)
                state_year_dict["avg_median_sale_price"] = avg_median_sale_price

                avg_homes_sold = state_year_dict.get("avg_homes_sold", [])
                avg_homes_sold.append(homes_sold)
                state_year_dict["avg_homes_sold"] = avg_homes_sold

                total_homes_sold = state_year_dict.get("total_homes_sold", 0)
                state_year_dict["total_homes_sold"] = total_homes_sold + homes_sold

                if property_type != "":
                    homes_type_sold = state_year_dict.get(property_type, 0)
                    state_year_dict[property_type] = homes_type_sold + homes_sold

                avg_new_listings = state_year_dict.get("avg_new_listings", [])
                avg_new_listings.append(new_listings)
                state_year_dict["avg_new_listings"] = avg_new_listings

                avg_median_dom = state_year_dict.get("avg_median_dom", [])
                avg_median_dom.append(median_dom)
                state_year_dict["avg_median_dom"] = avg_median_dom

                state_dict[start_date.year] = state_year_dict
                self.data_points[state] = state_dict

        self.data = []
        self.targets = []

        years = range(2012, 2022)
        for state in self.data_points:
            state_data_tensor = None
            state_target_tensor = None
            for year in years:
                state_year_dict = self.data_points[state][year]
                avg_median_ppsf = state_year_dict.get("avg_median_ppsf", [0.0])
                avg_median_ppsf = sum(avg_median_ppsf)
                avg_median_list_ppsf = state_year_dict.get(
                    "avg_median_list_ppsf", [0.0])
                avg_median_list_ppsf = sum(avg_median_list_ppsf)
                avg_median_sale_price = state_year_dict.get(
                    "avg_median_sale_price", [0.0])
                avg_median_sale_price = sum(avg_median_sale_price)
                avg_homes_sold = state_year_dict.get("avg_homes_sold", [0.0])
                avg_homes_sold = sum(avg_homes_sold)
                total_homes_sold = state_year_dict.get("total_homes_sold", 0)
                avg_new_listings = state_year_dict.get(
                    "avg_new_listings", [0.0])
                avg_new_listings = sum(avg_new_listings)
                avg_median_dom = state_year_dict.get("avg_median_dom", [0.0])
                avg_median_dom = sum(avg_median_dom)
                single_fam_res = state_year_dict.get(
                    "Single Family Residential", 0)
                townhouse = state_year_dict.get("Townhouse", 0)
                condo = state_year_dict.get("Condo/Co-op", 0)
                all_res = state_year_dict.get("All Residential", 0)
                multi_fam = state_year_dict.get("Multi-Family (2-4 Unit)", 0)

                year_tensor = torch.tensor([[avg_median_ppsf, avg_median_list_ppsf, avg_median_sale_price, avg_homes_sold,
                                           total_homes_sold, avg_new_listings, avg_median_dom, single_fam_res, townhouse, condo, all_res, multi_fam]])
                if state_data_tensor is None:
                    state_data_tensor = year_tensor
                else:
                    state_data_tensor = torch.cat(
                        (state_data_tensor, year_tensor), dim=0)

                next_year_dict = self.data_points[state].get(year + 1)
                if next_year_dict is not None:
                    next_avg_ppsf = next_year_dict.get(
                        "avg_median_ppsf", [0.0])
                    next_avg_ppsf = sum(next_avg_ppsf)
                    target_tensor = torch.tensor([next_avg_ppsf])
                else:
                    target_tensor = torch.tensor([0.0])

                if state_target_tensor is None:
                    state_target_tensor = target_tensor
                else:
                    state_target_tensor = torch.cat(
                        (state_target_tensor, target_tensor), dim=0)

            self.data.append(state_data_tensor)
            self.targets.append(state_target_tensor)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
