library("proj4")
library("spdep")
library("maptools")
library("rgdal")
library('dplyr')

#script.dir <- dirname(sys.frame(1)$ofile)
#setwd(script.dir)

smoothed = function(k, filename) {
  shape_file = readOGR('data/misc/us_county_hs_only', "us_county_hs_only")
  
  selection_ratio_df = read.csv("data/output/selection_ratio_county_2019.csv", row.names = 1, colClasses=c("FIPS"="character"))
  selection_ratio_df = selection_ratio_df %>% select("selection_ratio", "FIPS")
  is.na(selection_ratio_df)<-sapply(selection_ratio_df, is.infinite)
  selection_ratio_df[is.na(selection_ratio_df)] = 0
  
  colnames(selection_ratio_df) = c("selection_ratio", "GEOID")
  mainland.states = c(1,4:6, 8:13, 16:42, 44:51, 53:56)
  shape_file = shape_file[shape_file$STATEFP %in% sprintf('%02i', mainland.states), ]
  # NOT SURE WHAT THE CORRECT PROCEDURE IS HERE...for now just remove counties that we done have?
  shape_file = shape_file[shape_file$GEOID %in% selection_ratio_df$GEOID, ]
  
  df <- droplevels(as(shape_file, 'data.frame'))
  
  aea.proj = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96"
  shape_file = spTransform(shape_file, CRS(aea.proj))  # project to Albers
  
  df = inner_join(selection_ratio_df, df, by = c('GEOID' = 'GEOID'))
  
  coords = coordinates(shape_file)
  IDs<-row.names(as(shape_file, "data.frame"))
  
  knn <- knn2nb(knearneigh(coords, k = k), row.names = IDs)
  knn <- include.self(knn)
  
  localGvalues <- localG(x = as.numeric(df$selection_ratio), listw = nb2listw(knn, style = "B"), zero.policy = TRUE)
  localGvalues <- round(localGvalues,3)
  
  new_df <- data.frame(df$GEOID)
  new_df$values <- localGvalues
  
  write.table(row.names = FALSE, new_df, file = filename, sep=",")
}

smoothed(k=10, filename = "test.csv")
