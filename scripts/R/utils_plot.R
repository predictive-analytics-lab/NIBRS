map_to_colors <- function(quartile_alpha, sr_binned){
  pos_quartile <- case_when(
    quartile_alpha == 0.25 ~ 0,
    quartile_alpha == 0.5 ~ 1,
    quartile_alpha == 0.75 ~ 2,
    quartile_alpha == 1 ~ 3
  )
  pos_sr_binned <- case_when(
    sr_binned == 'S<0.8' ~ 4 * 4 + 1,
    sr_binned == '0.8\u2264 S < 1.25' ~ 4 * 3 + 1,
    sr_binned == '1.25\u2264 S < 2' ~ 4 * 2 + 1,
    sr_binned == '2\u2264 S < 5' ~ 4 + 1,
    sr_binned == 'S \u2265 5' ~ 1
  )
  code <- pos_quartile + pos_sr_binned
  return(letters[code])
}

assign_quartile <- function(x, quartiles){
  x <- case_when(
    x > quartiles[3] ~ 4,
    x > quartiles[2] ~ 3,
    x > quartiles[1] ~ 2,
    x <= quartiles[1] ~ 1
  )
  x / 4
}