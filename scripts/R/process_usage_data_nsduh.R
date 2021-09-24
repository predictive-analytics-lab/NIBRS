renv::restore()

args <- commandArgs(trailingOnly = TRUE)


if(length(args)!=6){
  poverty <- 0
  metro <- 0
  hispanics_included <- 0
  aggregate <- 0
  year_min <- 2010
  year_max <- 2019
} else{
  print(args)
  poverty <- ifelse(args[1]==1, 1, 0)
  metro <- ifelse(args[2]==1, 1, 0)
  hispanics_included <- ifelse(args[3]==1, 1, 0)
  aggregate <- ifelse(args[4]==1, 1, 0)
  year_min <- ifelse(!is.na(args[5]), args[5], 2010)
  year_max <- ifelse(!is.na(args[6]), args[6], 2019)
}

library(dplyr)
library(purrr)
library(readr)
library(vroom)
library(glue)
library(here)
library(srvyr)
library(missRanger)
library(cli)
library(tidyr)
library(ranger)


cli_h1('Processing NSDUH files!')
cli_text('Parameters')
cli_li(glue('poverty: {poverty}'))
cli_li(glue('metro: {metro}'))
cli_li(glue('hispanic included: {hispanics_included}'))
cli_li(glue('aggregate: {aggregate}'))
cli_li(glue('years: {year_min}-{year_max}'))
cli_h1('')

# download and store all files ----

years <- year_min:year_max
get_nsduh_data <- function(year){
    download.file(url = glue('https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-{year}/NSDUH-{year}-datasets/NSDUH-{year}-DS0001/NSDUH-{year}-DS0001-bundles-with-study-info/NSDUH-{year}-DS0001-bndl-data-tsv.zip'),
                  destfile = here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
    #dir.create(here('scripts', 'R', 'downloaded_data', 'nsduh'))
    unzip(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'),
          exdir = here('scripts', 'R', 'downloaded_data', glue('nsduh_{year}')))
    file.remove(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
}

# if filename is already present, just skip everything
filename_to_write <- here('scripts', 'R', 'downloaded_data', 'nsduh', glue('nsduh_usage_{ifelse(aggregate==1, "aggregate_", "")}{min(years)}_{max(years)}{ifelse(hispanics_included == 1, "_hispincluded", "_nohisp")}{ifelse(poverty==1, "_poverty", "")}{ifelse(metro==1, "_metro", "")}.csv'))
# if(file.exists(filename_to_write)) q()


# download only files that if they are not present
dir.create(here('scripts', 'R', 'downloaded_data'))
files_present <- list.files(here('scripts', 'R', 'downloaded_data'))
years_to_download <- years[!(glue('nsduh_{years}') %in% files_present)]

if(length(years_to_download)>0){
  years %>%
    map( ~ .x %>% get_nsduh_data(.))
}


# load file of interest ----

# merge all data files

df_list <- years %>% map(~ vroom(here('scripts', 'R', 'downloaded_data',
                           glue('nsduh_{.x}'), glue('NSDUH_{.x}_Tab.{if(.x==2019){"txt"}else{"tsv"}}')),
                           col_types = cols()) %>%
                           mutate(VEREP = as.character(VEREP)))
names(df_list) <- years

# keep only categories that are needed for the analysis
# codebook https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2002-2019/NSDUH-2002-2019-datasets/NSDUH-2002-2019-DS0001/NSDUH-2002-2019-DS0001-info/NSDUH-2002-2019-DS0001-info-codebook.pdf
cols_to_keep <- c(
                  "CATAG7",
                  "CATAG6",
                  "MJEVER",
                  "MJAGE",
                  "IRSEX",
                  "COUTYP2",
                  "MJDAY30A",
                  "ANALWT_C",
                  "NEWRACE2", # race/hispanic ethnicity recode
                  "PDEN10", # POPULATION DENSITY 2010
                  "COUTYP4", # COUNTY metro/nonmetro STATUS
                  "IRPINC3", # RECODE -RESP TOT INCOME - IMPUTATION REVISED, from 2015 onwards
                  "IRFAMIN3", # RECODE - TOT FAM INCOME - IMPUTATION REVISED
                  "POVERTY3", # RC-POVERTY LEVEL (% OF US CENSUS POVERTY THRESHOLD)
                  "POVERTY2",
                  "MMBT30DY",
                  "MMT30FRQ",
                  "MMGETMJ",
                  "MMLSLBS",
                  "MMLS10GM",
                  "MMLSUNIT",
                  "MMLSOZS",
                  "MMLSGMS",
                  "MMBPLACE",
                  "MMBUYWHO",
                  "ANALWT_C",
                  "MMBPLACE",
                  "MMBCLOSE",
                  "MMLSPCTB",
                  
                  "BKDRVINF", # ARRESTED AND BOOKED
                  # USE OF ALCOHOL/SUBSTANCES
                  "DRVINALCO2",
                  "DRVINMARJ2",
                  "DRVINDRG",
                  "DRVINDROTMJ",
                  "DRVINALDRG", 
                  "DRVALDR",
                  "DRVDONLY",
                  "DRVAONLY",
                  # "DRVINALCO",
                  # "DRVINMARJ",
                  # "DRVINCOCN",
                  # "DRVINHERN",
                  # "DRVINHALL",
                  # "DRVININHL",
                  # "DRVINMETH",
                  # # from 2016 onwards
                  # "DRVINALCO2",
                  # "DRVINMARJ2",
                  # "DRVINMDRG",
                  
                  "VEREP",
                  "VESTR" # stratum
)


df <- df_list %>% map_dfr(~ .x %>% select(any_of(cols_to_keep)),
                          .id = 'year')

# process data ----

df <- df %>% 
  mutate(
    sex = case_when(
      IRSEX == 1 ~ 'male',
      IRSEX == 2~ 'female'
    ),
    # age = case_when(
    #   CATAG7 == 1 ~ '12-13',
    #   CATAG7 == 2 ~ '14-15',
    #   CATAG7 == 3 ~ '16-17',
    #   CATAG7 == 4 ~ '18-20',
    #   CATAG7 == 5 ~ '21-25',
    #   CATAG7 == 6 ~ '26-34',
    #   CATAG6 == 4 ~ '35-49',
    #   CATAG6 == 5 ~ '50-64',
    #   CATAG6 == 6 ~ '65+'
    # ),
    age = case_when(
      CATAG7 == 1 ~ '12-17',
      CATAG7 == 2 ~ '12-17',
      CATAG7 == 3 ~ '12-17',
      CATAG7 == 4 ~ '18-25',
      CATAG7 == 5 ~ '18-25',
      CATAG7 == 6 ~ '26-34',
      CATAG6 == 4 ~ '35-49',
      CATAG6 == 5 ~ '50+',
      CATAG6 == 6 ~ '50+'
    ),
    poverty_level = case_when(
      POVERTY3 == 1 | POVERTY2 == 1 ~ 'living in poverty',
      POVERTY3 == 2 | POVERTY3 == 3 | POVERTY2 == 2 | POVERTY2 == 3 ~ 'income higher than poverty threshold',
    ),
    # look for nonmetro vs. metro here OMB vs. RUCA and recoding done here
    # https://www.hrsa.gov/rural-health/about-us/definition/index.html
    is_metro = case_when(
      COUTYP4 == 1 | COUTYP2 == 1~ 'metro',
      COUTYP4 == 2 | COUTYP2 == 2 ~ 'metro',
      COUTYP4 == 3 | COUTYP2 == 3 ~ 'nonmetro'
    ),
    usage_days = case_when(
      MJDAY30A <= 30 ~ MJDAY30A,
      MJDAY30A == 91 | MJDAY30A == 93 ~ 0
      # the remaining codes categorized as NAs
    ),
    usage_ever = case_when(
      MJEVER == 1 ~ 1,
      MJEVER == 2 ~ 0,
    ),
    usage_agefirsttime = case_when(
      MJAGE >= 1 & MJAGE <= 82 ~ MJAGE,
      MJAGE == 991 ~ 0
    ),
    days_bought_marijuana = case_when(
      MMBT30DY >= 1 & MMBT30DY <= 30 ~ MMBT30DY,
      MMBT30DY == 91 | MMBT30DY == 93 | MMBT30DY == 99 ~ 0
    ),
    days_traded_marijuana = case_when(
      MMT30FRQ >= 1 & MMT30FRQ <= 30 ~ MMT30FRQ,
      MMT30FRQ == 91 | MMT30FRQ == 93 | MMT30FRQ == 99 ~ 0
    ),
    days_bought_marijuana_outside = case_when(
      MMBT30DY >= 1 & MMBT30DY <= 30 & MMBPLACE != 4 ~ MMBT30DY,
      MMBT30DY == 91 | MMBT30DY == 93 | MMBT30DY == 99 ~ 0
    ),
    days_traded_marijuana_outside = case_when(
      MMT30FRQ >= 1 & MMT30FRQ <= 30 & MMBPLACE != 4 ~ MMBT30DY,
      MMT30FRQ == 91 | MMT30FRQ == 93 | MMT30FRQ == 99 ~ 0
    ),
    dui_past_year = case_when(
      DRVINALCO2 == 1 | 
        DRVINMARJ2 == 1 | 
        DRVINDRG == 1 | 
        DRVINDROTMJ == 1 | 
        DRVINALDRG == 1 | 
        DRVALDR == 1  | 
        DRVAONLY == 1 | 
        DRVDONLY == 1 ~ 1,
      DRVINALCO2 == 0 | 
        DRVINMARJ2 == 0 | 
        DRVINDRG == 0 | #DRVINDRG == 91 | DRVINDRG == 81 | DRVINDRG == 99 |
        DRVINDROTMJ == 0 | #DRVINDROTMJ == 91 | DRVINDROTMJ == 81 | DRVINDROTMJ == 99 |
        DRVINALDRG == 0 | #DRVINALDRG == 91 | DRVINALDRG == 81 |  DRVINALDRG == 99 |
        DRVALDR == 2  | DRVALDR == 81 | DRVALDR == 91 | DRVALDR == 99 |
        DRVAONLY == 2 | DRVAONLY == 81 | DRVAONLY == 91 | DRVAONLY == 99 |
        DRVDONLY == 2 | DRVDONLY == 81 | DRVDONLY == 91 | DRVDONLY == 99 ~ 0
    )
  )



if(hispanics_included == 1){

  # impute missing data
  imputed_df <- missRanger(df %>% select(NEWRACE2, IRFAMIN3, sex, age, usage_ever, poverty_level, is_metro, usage_agefirsttime), pmm.k = 3, num.trees = 100)

  # fit rf
  rf <- ranger(is_white ~ sex + age + usage_ever + IRFAMIN3 + poverty_level + poverty_level + usage_agefirsttime,
               data = imputed_df %>% mutate(is_white = case_when(
                 NEWRACE2 == 1 ~ 1,
                 NEWRACE2 == 2 ~ 0
               )) %>%
                 mutate(is_white = as.factor(is_white)) %>%
                 filter(!is.na(is_white)),
               probability = TRUE
  )
  pred_is_white <- predict(rf, imputed_df %>% filter(NEWRACE2 == 7))$predictions
  # we can't really predict race well, but going with this for the moment
  col_to_use <- ifelse(mean(pred_is_white[,1]>0.5)>0.5, 1, 2)
  pred_is_white <- pred_is_white[,col_to_use]

  # randomly sample which race the person belongs to
  df[df$NEWRACE2 == 7,'NEWRACE2'] <- ifelse(runif(length(pred_is_white), 0, 1) < pred_is_white, 1, 2)
}


df <- df %>%
  mutate(race = case_when(
    NEWRACE2 == 1 ~ 'white',
    NEWRACE2 == 2 ~ 'black',
    NEWRACE2 >= 3 ~ 'other'
  ))

# transform into survey data syntax http://gdfe.co/srvyr/
# https://www.jstatsoft.org/article/view/v009i08
df_srv <- df %>%
  #mutate(year_strata = glue('{year}{VESTR}')) %>%
  as_survey_design(strata = VESTR, ids = VEREP, weights = ANALWT_C,
                   nest = TRUE)

vars_group <- c('race', 'sex', 'age')
if(aggregate!=1) vars_group <- c('year', vars_group)
if(poverty == 1)  vars_group <- c(vars_group, 'poverty_level')
if(metro == 1)  vars_group <- c(vars_group, 'is_metro')


# TODO: look at how variance is computed without the replicate weights
stats_df <- df_srv %>%
   group_by(across(all_of(vars_group))) %>%
   summarise(
     ever_used = survey_mean(usage_ever, na.rm = TRUE),
     mean_usage_day = survey_mean(usage_days/30, na.rm = TRUE),
     mean_bought_day = survey_mean(days_bought_marijuana/30, na.rm = TRUE),
     mean_bought_outside_day = survey_mean(days_bought_marijuana_outside/30,
                                            na.rm = TRUE),
     mean_traded_day = survey_mean(days_traded_marijuana/30,
                                    na.rm = TRUE),
     mean_traded_outside_day = survey_mean(days_traded_marijuana_outside/30,
                                            na.rm = TRUE),
     dui_past_year = survey_mean(dui_past_year, na.rm = TRUE)
   )

dir.create(here('scripts', 'R', 'downloaded_data', 'nsduh'))

stats_df %>%
  write_csv(filename_to_write)



# generate summary stats for table ----

# df4stats <- df %>%
#   filter(race != 'other') %>%
#   mutate(
#     MMGETMJ_label = case_when(
#       MMGETMJ == 1 ~ 'Bought it',
#       MMGETMJ == 2 ~ 'Traded for it',
#       MMGETMJ == 3 ~ 'Got it for free',
#       MMGETMJ == 4 ~ 'Grew it',
#       MMGETMJ == 91 ~ 'Never used',
#       MMGETMJ == 93 ~ 'Not used in past year'
#     ),
#     buying_frequency = case_when(
#       MMBT30DY >= 1 & MMBT30DY <= 30 ~ MMBT30DY,
#       MMBT30DY == 91 | MMBT30DY == 93 |  MMBT30DY == 99 ~ 0,
#     ),
#     how_much_marijuana_last_time = case_when(
#       MMLSGMS == 1 | MMLSGMS == 2 | MMLSOZS == 1 | MMLSOZS ==2 ~ '<10 grams',
#       MMLSGMS == 3 | MMLSOZS == 3 | MMLSOZS == 4 | MMLSOZS == 5 | MMLSOZS == 6 | MMLSOZS == 7 ~ '>10 grams',
#       MMLSGMS == 91 | MMLSGMS == 99 | MMLSGMS == 93 | MMLSOZS == 91 | MMLSOZS == 93 | MMLSOZS == 99 ~ 'Not used/skip'),
#     how_much_paid = case_when(
#       MMLSPCTB <= 3 ~ '<20$',
#       MMLSPCTB <= 4 ~ '21-50$',
#       MMLSPCTB <= 5 ~ '51-100$',
#       MMLSPCTB > 5 & MMLSPCTB <= 12 ~ '>100$',
#       MMLSPCTB == 91 |  MMLSPCTB == 93 ~ 'Not used/skip'),
#     who_sold_you_marijuana_last_time = case_when(
#       MMBUYWHO == 1 ~ 'Friend',
#       MMBUYWHO == 2 ~ 'Relative',
#       MMBUYWHO == 3 ~ 'Stranger',
#       MMBUYWHO == 91 | MMBUYWHO == 93 | MMBUYWHO == 99 ~ 'Not used/skip'
#     ),
#     where_did_you_buy = case_when(
#       MMBPLACE == 1 | MMBPLACE == 11~ 'Inside public building',
#       MMBPLACE == 2 | MMBPLACE == 3 | MMBPLACE == 13 ~ 'At school',
#       MMBPLACE == 4 | MMBPLACE == 14 ~ 'Inside a house',
#       MMBPLACE == 5 | MMBPLACE == 15 ~ 'Outside in public area',
#       MMBPLACE == 6 ~ 'Other',
#       MMBPLACE == 91 | MMBPLACE == 93 | MMBPLACE == 99 ~ 'Not used/skip'),
#     how_near_were_you_when_bought_marijuana_last_time = case_when(
#       MMBCLOSE == 1 ~ 'Near home',
#       MMBCLOSE == 2 ~ 'Somewhere else',
#       MMBCLOSE == 91 | MMBCLOSE == 93 | MMBCLOSE == 99 ~ 'Not used/skip'
#     )
#     )
# 
# df4stats %>% group_by(year) %>% summarise(mean(is.na(MMGETMJ_label)))
# df4stats %>% group_by(year) %>% summarise(mean(is.na(buying_frequency)))
# df4stats %>% group_by(year) %>% summarise(mean(is.na(how_much_marijuana_last_time)))
# df4stats %>% group_by(year) %>% summarise(mean(is.na(how_much_paid)))
# df4stats %>% group_by(year) %>% summarise(mean(is.na(how_near_were_you_when_bought_marijuana_last_time)))
# 
# get_how_get_marijuana <- function(grouping=FALSE, df4stats_srv){
#   #browser()
#   if(grouping){
#     to_group <- c(grouping, 'MMGETMJ_label')
#   } else{
#     to_group <- 'MMGETMJ_label'
#   }
#   how_get_last_marijuana_used <- df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(MMGETMJ_label != 'Never used' & MMGETMJ_label != 'Not used in past year') %>%
#     group_by(across(all_of(to_group))) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     #filter(MMGETMJ_label != 'Never used' & MMGETMJ_label != 'Not used in past year') %>%
#     mutate(value = glue('{round(value,2) * 100}% ({round(value_se,2)*100})'))
#   #how_get_last_marijuana_used %>% select(-value_se) %>%
#   #  pivot_wider(names_from = race, values_from = value)
#   how_get_last_marijuana_used %>%
#     select(MMGETMJ_label, value) %>%
#     rename(term = MMGETMJ_label)
# }
# 
# get_buying_frequency <- function(df4stats_srv){
# 
#   # among those that actually do buy
#   buy <- df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(buying_frequency > 0 & !is.na(buying_frequency)) %>%
#     summarise(
#       value = survey_mean(buying_frequency)
#     ) %>%
#     mutate(value = glue('{value %>% round()} ({value_se %>% round(.)})'), 
#            value_se = glue('{value_se %>% round(.)}'))
#   perc_buyers <- df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     summarise(
#       value = survey_mean(buying_frequency>0, na.rm = TRUE)
#     ) %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({round(value_se, 2) * 100})'), 
#            value_se = glue('{value_se %>% round(.,2) * 100}%'))
#   buy %>%
#     mutate(term = "Mean number of days among buyers") %>%
#     bind_rows(
#       perc_buyers %>% mutate(term = "% buyers")
#     )
# }
# 
# get_how_much_marijuana_last_time <- function(df4stats_srv){
#   df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(how_much_marijuana_last_time != 'Not used/skip') %>%
#     group_by(how_much_marijuana_last_time) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     # filter(how_much_marijuana_last_time != 'Not used/skip') %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({round(value_se,2) * 100})'), 
#            value_se = glue('{value_se %>% round(.,2) * 100}%')) %>%
#     rename(term = how_much_marijuana_last_time)
# }
# 
# 
# get_how_much_paid_last_time <- function(df4stats_srv){
#   stats <- df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(how_much_paid != 'Not used/skip')  %>%
#     group_by(how_much_paid) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({round(value_se,2) * 100})'), 
#            value_se = glue('{value_se %>% round(.,2) * 100}%')) %>%
#     rename(term = how_much_paid)
#   stats[c(1,3,4,2),]
# }
# 
# get_who_sold_you_marijuana_last_time <- function(df4stats_srv){
#   df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(who_sold_you_marijuana_last_time != 'Not used/skip')  %>%
#     group_by(who_sold_you_marijuana_last_time) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     #filter(who_sold_you_marijuana_last_time != 'Not used/skip')  %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({round(value_se, 2)*100})'), 
#            value_se = glue('{round(value_se,2) * 100}%')) %>%
#     rename(term = who_sold_you_marijuana_last_time)
# }
# 
# get_where_did_you_buy <- function(df4stats_srv){
#   df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(where_did_you_buy != 'Not used/skip')  %>%
#     group_by(where_did_you_buy) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     #filter(where_did_you_buy != 'Not used/skip')  %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({value_se %>% round(.,2) * 100})'), 
#            value_se = glue('{value_se %>% round(.,2) * 100}%')) %>%
#     rename(term = where_did_you_buy)
# }
# 
# get_how_near_were_you_when_bought_marijuana_last_time <- function(df4stats_srv){
#   df4stats_srv %>%
#     filter(!(year %in% c(2015, 2016, 2017))) %>%
#     filter(how_near_were_you_when_bought_marijuana_last_time != 'Not used/skip') %>%
#     group_by(how_near_were_you_when_bought_marijuana_last_time) %>%
#     summarise(
#       value = survey_mean()
#     ) %>%
#     mutate(value = glue('{value %>% round(.,2) * 100}% ({value_se %>% round(.,2) * 100})'), 
#            value_se = glue('{value_se %>% round(.,2) * 100}%')) %>%
#     #filter(how_near_were_you_when_bought_marijuana_last_time != 'Not used/skip') %>%
#     rename(term = how_near_were_you_when_bought_marijuana_last_time)
# }
# 
# 
# df4stats_srv <- df4stats %>% 
#   as_survey_design(strata = VESTR, weights = ANALWT_C)
# df4stats_list <- list(
#   df4stats_srv %>% filter(race == 'black' & is_metro == 'metro'),
#   df4stats_srv %>% filter(race == 'black' & sex == 'male' & age == '18-25' & is_metro == 'metro'),
#   df4stats_srv %>% filter(race == 'white' & is_metro == 'metro'),
#   df4stats_srv %>% filter(race == 'white' & sex == 'male' & age == '18-25' & is_metro == 'metro'),
#   df4stats_srv %>% filter(race == 'black' & is_metro == 'nonmetro'),
#   df4stats_srv %>% filter(race == 'black' & sex == 'male' & age == '18-25' & is_metro == 'nonmetro'),
#   df4stats_srv %>% filter(race == 'white' & is_metro == 'nonmetro'),
#   df4stats_srv %>% filter(race == 'white' & sex == 'male' & age == '18-25' & is_metro == 'nonmetro')
# )
#   
# ## 
# tables_stats <- list(
#   "How did you get marijuana?" = df4stats_list %>%
#   map(~ get_how_get_marijuana(FALSE, .x)) %>%
#   reduce(left_join, by = 'term'),
#   
#   "How often did you buy it?" = df4stats_list %>%
#     map(~ get_buying_frequency(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How much marijuana did you get last time?" = df4stats_list %>%
#     map(~ get_how_much_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#    "How much did you for marijuana last time?" = df4stats_list %>%
#     map(~ get_how_much_paid_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "Who sold you marijuana last time?" = df4stats_list %>%
#     map(~ get_who_sold_you_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "Where did you buy marijuana last time?" = df4stats_list %>%
#     map(~ get_where_did_you_buy(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How near you when you bought marijuana last time?" = df4stats_list %>%
#     map(~ get_how_near_were_you_when_bought_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term')
# )
# 
# print(tables_stats %>%
#   bind_rows(.id = 'variable') %>%
#   xtable(., 
#          align = rep('c', 11)),
#   include.rownames=FALSE)
# 
# 
# 
# 
# ## poverty
# 
# df4stats_srv <- df4stats %>% 
#   as_survey_design(strata = VESTR, weights = ANALWT_C)
# df4stats_list <- list(
#   df4stats_srv %>% filter(race == 'black' & poverty_level == 'income higher than poverty threshold'),
#   df4stats_srv %>% filter(race == 'black' & sex == 'male' & age == '18-25' & poverty_level == 'income higher than poverty threshold'),
#   df4stats_srv %>% filter(race == 'white' & poverty_level == 'income higher than poverty threshold'),
#   df4stats_srv %>% filter(race == 'white' & sex == 'male' & age == '18-25' & poverty_level == 'income higher than poverty threshold'),
#   df4stats_srv %>% filter(race == 'black' & poverty_level == 'living in poverty'),
#   df4stats_srv %>% filter(race == 'black' & sex == 'male' & age == '18-25' & poverty_level == 'living in poverty'),
#   df4stats_srv %>% filter(race == 'white' & poverty_level == 'living in poverty'),
#   df4stats_srv %>% filter(race == 'white' & sex == 'male' & age == '18-25' & poverty_level == 'living in poverty')
# )
# 
# ## 
# tables_stats <- list(
#   "How did you get marijuana?" = df4stats_list %>%
#     map(~ get_how_get_marijuana(FALSE, .x)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How often did you buy it?" = df4stats_list %>%
#     map(~ get_buying_frequency(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How much marijuana did you get last time?" = df4stats_list %>%
#     map(~ get_how_much_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How much did you for marijuana last time?" = df4stats_list %>%
#     map(~ get_how_much_paid_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "Who sold you marijuana last time?" = df4stats_list %>%
#     map(~ get_who_sold_you_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "Where did you buy marijuana last time?" = df4stats_list %>%
#     map(~ get_where_did_you_buy(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term'),
#   
#   "How near you when you bought marijuana last time?" = df4stats_list %>%
#     map(~ get_how_near_were_you_when_bought_marijuana_last_time(.x) %>%
#           select(term, value)) %>%
#     reduce(left_join, by = 'term')
# )
# 
# print(tables_stats %>%
#         bind_rows(.id = 'variable') %>%
#         xtable(., 
#                align = rep('c', 11)),
#       include.rownames=FALSE)
# 
