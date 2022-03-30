library(stats)
library(here)
library(vroom)
library(reshape2)
library(stringr)
library(stargazer)
library(gtools)
library(xtable)

renv::restore()

popdf = vroom("data/output/selection_ratio_county_2019_bootstraps_1000_poverty.csv") %>%
  select(FIPS, black, white) %>%
  rename(FIPS = FIPS, black_pop = black, white_pop = white)

usage = vroom("data/output/selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv") %>% 
  select(FIPS, black_incidents, white_incidents, white_users, black_users, black, white) %>% 
  rename(white_usage = white_users, black_usage = black_users, black_pop = black, white_pop = white)

buying = vroom("data/output/selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv") %>% 
  select(FIPS, white_users, black_users, black_incidents, white_incidents,black, white) %>% 
  rename(white_buyers = white_users, black_buyers = black_users, black_incidents=black_incidents, white_incidents=white_incidents,black_pop = black, white_pop = white)

buying_outside = vroom("data/output/selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv") %>% 
  select(FIPS, white_users, black_users) %>% 
  rename(white_outside_buyers = white_users, black_outside_buyers = black_users)

# data = usage %>% 
#   left_join(buying, by=c("FIPS" = "FIPS")) %>%
#   left_join(buying_outside, by=c("FIPS" = "FIPS"))
# 
# pop = data %>%
#   select(FIPS, black_pop, white_pop) %>% 
#   melt(data, id.vars=c("FIPS"), variable.name="race", measure.vars = c("black_pop", "white_pop"), value.name="population") %>% 
#   mutate(race = str_split(race, "_", simplify = TRUE)[, 1])
# 
# incidents = data %>% 
#   select(FIPS, black_incidents, white_incidents) %>% 
#   melt(data, id.vars=c("FIPS"), variable.name="race", measure.vars = c("black_incidents", "white_incidents"), value.name="incidents") %>% 
#   mutate(race = str_split(race, "_", simplify = TRUE)[, 1])
# 
# purchases = data %>% select(FIPS, black_buyers, white_buyers) %>% 
#   melt(data, id.vars=c("FIPS"), variable.name="race", measure.vars = c("black_buyers", "white_buyers"), value.name="purchases") %>% 
#   mutate(race = str_split(race, "_", simplify = TRUE)[, 1])
# 
# purchases_outside = data %>% select(FIPS, black_outside_buyers, white_outside_buyers) %>% 
#   melt(data, id.vars=c("FIPS"), variable.name="race", measure.vars = c("black_outside_buyers", "white_outside_buyers"), value.name="outside_purchases") %>% 
#   mutate(race = str_split(race, "_", simplify = TRUE)[, 1])
# 
# uses = data %>% select(FIPS, black_usage, white_usage) %>%
#   melt(data, id.vars=c("FIPS"), variable.name="race", measure.vars = c("black_usage", "white_usage"), value.name="uses") %>%
#   mutate(race = str_split(race, "_", simplify = TRUE)[, 1])

# joined = incidents %>% left_join(purchases, by=c("FIPS" = "FIPS", "race" = "race")) %>% left_join(purchases_outside, by=c("FIPS" = "FIPS", "race" = "race")) %>% left_join(uses, by=c("FIPS" = "FIPS", "race" = "race")) %>% left_join(pop, by=c("FIPS" = "FIPS", "race" = "race"))
# 
# joined$incidents = as.integer(joined$incidents)
# joined$uses = as.integer(joined$uses)
# joined$purchases = as.integer(joined$purchases)
# joined$outside_purchases = as.integer(joined$outside_purchases)
# 
# joined$population = joined$population + 1
# joined$uses = joined$uses + 1
# joined$purchases = joined$purchases + 1
# joined$outside_purchases = joined$outside_purchases + 1
# 
# joined$usespp = joined$uses / joined$population
# joined$purchasespp = joined$purchases / joined$population
# joined$purchases_outsidepp = joined$outside_purchases / joined$population
# 
# joined$incidentspp = joined$incidents / joined$population

data = usage %>% left_join(buying, by=c("FIPS" = "FIPS", "black_incidents" = "black_incidents", "white_incidents" = "white_incidents")) %>% left_join(buying_outside, by=c("FIPS" = "FIPS")) %>% group_by(FIPS) %>% 
  summarise(black_outside_buyers=sum(black_outside_buyers), white_outside_buyers=sum(white_outside_buyers), black_incidents=sum(black_incidents), white_incidents=sum(white_incidents), black_users=sum(black_usage), white_users=sum(white_usage), black_buyers=sum(black_buyers), white_buyers=sum(white_buyers)) %>%
  mutate(purchases_outside=black_outside_buyers+white_outside_buyers, incidents = black_incidents + white_incidents, users = white_users + black_users, purchases=black_buyers+white_buyers) %>%
  left_join(popdf, by=c("FIPS" = "FIPS")) %>%
  # mutate(black_pop = replace_na(black_pop, 0), white_pop = replace_na(white_pop, 0)) %>%
  mutate(population = black_pop + white_pop) %>%
  drop_na(population, black_pop, white_pop) %>%
  filter(population != Inf, black_pop != 0, white_pop != 0) %>%
  mutate(incidentspp = incidents / population, purchases_outsidepp = purchases_outside / population, purchasespp = purchases / population, userspp = users / population) %>%
  mutate(black_buyingpp = black_buyers / black_pop, white_buyingpp = white_buyers / white_pop, black_buyingoutsidepp = black_outside_buyers / black_pop, white_buyingoutsidepp = white_outside_buyers / white_pop, black_userspp = black_users / black_pop, white_userspp = white_users / white_pop) %>%
  mutate(white_incidentspp = white_incidents / white_pop, black_incidentspp = black_incidents / black_pop)


# All Races
m1 = lm(incidentspp ~ userspp, data=data)
m2 = lm(incidentspp ~ purchasespp, data=data)
m3 = lm(incidentspp ~ purchases_outsidepp, data=data)

# Black
m4 = lm(black_incidentspp ~ black_userspp, data=data)
m5 = lm(black_incidentspp ~ black_buyingpp, data=data)
m6 = lm(black_incidentspp ~ black_buyingoutsidepp, data=data)

# White
m7 = lm(white_incidentspp ~ white_userspp, data=data)
m8 = lm(white_incidentspp ~ white_buyingpp, data=data)
m9 = lm(white_incidentspp ~ white_buyingoutsidepp, data=data)

format.result = function(model) {
  res = tidy(model)
  est = res[-1, 2]
  err = res[-1, 3]
  p.value = res[-1, 5]
  stars = stars.pval(p.value$p.value)
  return (paste0(signif(est, 2), " (", signif(err, 2),  ") ", stars))
}

headers = c("Race r", "P(Usage|Race=r)", "P(Purchasing|Race=r)", "P(Purchasing Outside|Race=r)")
r1 = c("All Incidents", format.result(m1), format.result(m2), format.result(m2))
r2 = c("Black", format.result(m4), format.result(m5), format.result(m6))
r3 = c("White", format.result(m7), format.result(m8), format.result(m9))
df = rbind(r1, r2, r3)
colnames(df) = headers
print(xtable(df, type = "latex"))
# model = glm(cbind(incidents, uses - incidents) ~ race*(usespp+purchasespp), family=binomial(), data=joined)
# stargazer::stargazer(model)
