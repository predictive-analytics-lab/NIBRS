library(here)
library(vroom)
library(ggplot2)
library(viridis)


source(here("scripts", "R", "utils_plot.R"))

cc <- vroom(here("data", "misc", "county_coverage.csv"))
cc <- cc %>% filter(year == 2019) %>% group_by(FIPS) %>% summarise(coverage = mean(coverage)) %>% mutate(coverage_binary = coverage > 0.8)
us_county <- map_data("county") %>%
  mutate(polyname = glue("{region},{subregion}")) %>%
  inner_join(county.fips, by = "polyname")
cc.county <- us_county  %>%
  left_join(cc %>%
              mutate(FIPS = as.numeric(FIPS)), by = c("fips" = "FIPS"))
p <- cc.county %>%
  ggplot(
    data = .,
    mapping = aes(
      x = long, y = lat,
      group = group,
      fill = coverage_binary
    )
  ) +
  geom_polygon(color = "white", size = 0.02) +
  theme_void() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  )
print(p)