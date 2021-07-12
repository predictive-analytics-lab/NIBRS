api = "1d38f929e06920fe9b0245a9d87968d98ce2bfd6"
Sys.setenv(CENSUS_KEY=api)


library(censusapi)


ages = c("<25", "25-44", "45-64", "65+")
income = c("<10K", "10-15K", "15-20K", "20-25K", "25-30K", "30-35K", "35-40K", "40-45K", "45-50K", "50-60K", "60-75K", "75-99K", "100-125K", "125-150K", "150-200K", "200K+")


white = paste0("B19037A_00", 3:69, "E")
black = paste0("B19037B_00", 3:69, "E")


getCensus(
  name = "acs/acs1", 
  vintage = 2019, 
  vars = c("NAME", white, black), 
  region = "county:*")