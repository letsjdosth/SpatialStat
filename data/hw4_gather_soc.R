library(soilDB)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

states = c(
"AL",	"KY",	"OH",
"LA",	"OK",
"AZ",	"ME",	"OR",
"AR",	"MD",	"PA",
"MA",
"CA", "MI",	"RI",
"CO", "MN",	"SC",
"CT", "MS",	"SD",
"DE", "MO",	"TN",
"MT",	"TX",
"FL", "NE",
"GA", "NV",	"UT",
"NH",	"VT",
"NJ",	"VA",
"ID", "NM",
"IL", "NY",	"WA",
"IN", "NC",	"WV",
"IA", "ND",	"WI",
"KS", "WY")
length(states)
# "AK", "AS", "PR", "DC", "TT", "GU", "HI","VI", "MP",

## Gather data
for(s in states){
  print(s)
  soc <- fetchRaCA(state=s) # select state here

  socSample <- soc$sample %>% 
    dplyr::select(sample_id, rcapid, soc, soc_sd, soc_measured, sample_top, sample_bottom, texture) ## carbon data
  socSite <- soc$pedons@site%>% 
    dplyr::select(rcapid, elevation=elev_field, long=x, lat=y, landuse) ## location data

  output <- socSample %>% left_join(socSite, by="rcapid") %>%
    filter(sample_top==0) %>% filter(soc_measured=="measured") ## just taking the top layer of soil
  head(output)
  write.csv(output, paste("data/soil_carbon_",s,".csv", sep=""))
  #write_csv(output, "soil_carbon.csv") ## Save data
}

## Map of data
md <- map_data("state", region="Pennsylvania")
ggplot()+
  geom_path(data=md, color="black", aes(x=long, y=lat, group=group))+
  geom_point(data=output, size=2, aes(x=long, y=lat, color=log(soc)))+
  scale_color_viridis_c()
  #theme_map()
