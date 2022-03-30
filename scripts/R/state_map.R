renv::restore()

library(dplyr)
library(purrr)
library(Cairo)
library(vroom)
library(here)
library(tigris)
library(ggplot2)
library(glue)
library(broom)
library(maps)
library(ggridges)
library(tidyr)
# library(cowplot)

source("../../scripts/R/utils_plot.R")

sr <- vroom(file.choose())
# sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv"))
# sr <- vroom(here("data", "output", "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv"))
sc <- fips_codes %>% mutate(FIPS = glue("{state_code}{county_code}"))

# figure 1 ----
cols <- c("#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#009E73")
opacity <- c("", "E6", "80", "4D")
all_cols <- cols %>%
  purrr::map(~ rev(paste0(.x, opacity))) %>%
  unlist()
all_cols <- tibble(cols = all_cols, code = letters[1:20])

sr <- sr %>%
  mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio)) %>%
  mutate(rel_err = 1 / ci)

sr = sr %>% filter(black_incidents > 0.5 & white_incidents > 0.5)

data("county.fips")
us_county <- map_data("county") %>%
  mutate(polyname = glue("{region},{subregion}")) %>%
  inner_join(county.fips, by = "polyname")

# get quartiles
quartiles <- sr %>%
  summarise(quant = quantile(
    rel_err,
    probs = c(0.25, 0.5, 0.75)
  )) %>%
  pull(quant)

sr <- sr %>%
  mutate(quartile_alpha = assign_quartile(
    rel_err,
    quartiles
  )) %>%
  mutate(sr_binned = case_when(
    selection_ratio < 0.8 ~ "S<0.8",
    selection_ratio >= 0.8 & selection_ratio < 1.25 ~ "0.8\u2264 S < 1.25",
    selection_ratio >= 1.25 & selection_ratio < 2 ~ "1.25\u2264 S < 2",
    selection_ratio >= 2 & selection_ratio < 5 ~ "2\u2264 S < 5",
    selection_ratio > 5 ~ "S \u2265 5"
  )) %>%
  mutate(
    color_code = factor(map_to_colors(quartile_alpha, sr_binned), levels = letters[1:20]) # %>%
  )

sr_x_county <- us_county %>%
  left_join(sr %>%
              mutate(FIPS = as.numeric(FIPS)), by = c("fips" = "FIPS"))

p <- sr_x_county %>%
  # bind_rows(sr_x_county) %>%
  # mutate(color_code = factor(color_code, levels = letters[1:20])) %>%
  ggplot(
    data = .,
    mapping = aes(
      x = long, y = lat,
      group = group,
      fill = color_code
    )
  ) +
  geom_polygon(color = "white", size = 0.1) +
  scale_fill_manual("Selection ratio S",
                    values = all_cols$cols,
                    labels = all_cols$code,
                    na.value = "gray90",
                    drop = FALSE
  ) +
  # theme_classic() +
  theme_void() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  ) +
  guides(fill = guide_legend("", ncol = 4, byrow = TRUE)) +
  theme(legend.text = element_blank())
print(p)