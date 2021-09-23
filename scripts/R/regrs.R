

df <- vroom('/Users/riccardofogliato/Downloads/processed_correlates.csv')


mod_fit <- lm(selection_ratio_dem_only ~ birthrate_bw_ratio, 
   data = df %>%
     mutate(selection_ratio_dem_only = 10^selection_ratio_log10_dem_only),
   weights = 1 / var_log_dem_only)
lmtest::coeftest(mod_fit, sandwich::sandwich(mod_fit))
