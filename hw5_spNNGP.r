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
names(df)
head(df)

library(spNNGP)

?spNNGP
starting_list = list("sigma.sq"=1, "tau.sq"=1, "phi"=1, "nu"=1)
tuning_list = list("sigma.sq"=0.5, "tau.sq"=0.5, "phi"=0.5, "nu"=0.5)
priors_list <- list("sigma.sq.IG"=c(0.01, 0.01), "tau.sq.IG"=c(0.01, 0.01), "phi.Unif"=c(0.01, 100), "nu.unif"=c(0.5,2.5))

fit_spnngp1 = spNNGP(
    soc~1, df, c("long","lat"),
    method = "response",
    n.neighbors=15,
    starting = starting_list,
    tuning = tuning_list,
    priors = priors_list,
    cov.model = "matern",
    n.samples = 100,
    return.neighbor.info=TRUE,
    n.omp.threads=3
)
summary(fit_spnngp1)

diag_spnngp1 = spDiag(fit_spnngp1)
diag_spnngp1

pred_spnngp1 <- predict(fit_spnngp1,
        X.0 = matrix(c(1),1,1),
        coords.0=matrix(c(36, 25), 1, 2)
)
pred_spnngp1$p.y.0 #response predictive samples


################################
?spConjNNGP

cov_param = c(phi=0.1, nu=1, alpha=1)

fit_conj_spnngp_1 = spConjNNGP(
    soc~1, df, c("long","lat"),
    n.neighbors=15,
    theta.alpha = cov_param,
    sigma.sq.IG=c(0.01, 0.01),
    cov.model="matern",
    k.fold=5,
    score.rule="crps",
    return.neighbor.info=TRUE,
    fit.rep =TRUE,
    n.samples=100,
    n.omp.threads=3
)
summary(fit_conj_spnngp_1)

diag_conj_spnngp_1 = spDiag(fit_conj_spnngp_1)
diag_conj_spnngp_1

pred_conj_spnngp_1 <- predict(fit_conj_spnngp_1,
        X.0 = matrix(c(1),1,1),
        coords.0=matrix(c(36, 25), 1, 2)
)
pred_fit_spnngp1$p.y.0 #response predictive samples

