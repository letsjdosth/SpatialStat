states = c("AL", "KY", "OH", "LA", "OK", "AZ", "ME", "OR", "AR", "MD", "PA", "MA",
"CA", "MI",	"RI", "CO", "MN", "SC", "CT", "MS",	"SD", "DE", "MO", "TN", "MT", "TX",
"FL", "NE", "GA", "NV",	"UT", "NH",	"VT", "NJ",	"VA", "ID", "NM", "IL", "NY", "WA",
"IN", "NC",	"WV", "IA", "ND", "WI", "KS", "WY")

df <- data.frame()
for(s in states){
df <- rbind(df, read.csv(paste("./data/soil_carbon_",s,".csv",sep="")))
}

df <- df[c("soc", "long", "lat")]
df$long <- df$long + rnorm(length(df$long)[1], 0, 0.01)
df$lat <- df$lat + rnorm(length(df$lat)[1], 0, 0.01)
dim(df)

library(INLA)

r = 10
coo = as.matrix(df[c("long", "lat")])
mesh = inla.mesh.2d(coo, cutoff=r/10, max.edge=c(r/4, r/2), offset=c(r/2, r))
plot(mesh, asp=1)
points(coo, col="red")

A <- inla.spde.make.A(mesh=mesh, loc=coo)
dim(A)
spde <- inla.spde2.pcmatern(
    mesh=mesh, alpha=1.5, 
    prior.range=c(0.1, 0.5),
    prior.sigma=c(100, 0.5)
)
stk <- inla.stack(
    tag="est",
    data=list(y=as.vector(df["soc"])),
    A=list(1, A),
    effect = list(data.frame(b0=1), df["long"], df["lat"])
)
?inla.stack
