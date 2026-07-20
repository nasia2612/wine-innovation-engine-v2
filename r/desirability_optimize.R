install.packages("desirability")
library(desirability)

range(design_coded$time)
range(design_coded$peak_rate)  
range(design_coded$conversion)


#desirability functions 
d_time <- dMin(low = min(design_coded$time), high = max(design_coded$time))
d_peak_rate<-dMin(low=min(design_coded$peak_rate),high=max(design_coded$peak_rate))
d_conv <- dMax(low =0.98, high = 0.99)#it is a constraint not optimization target

predict(d_time, 811.91)  
predict(d_time, 106.55)  
predict(d_conv, 0.8267)  
predict(d_conv, 1.0)      
predict(d_peak_rate, 4.0315215) #we dont want it high because it needs more cooling


#the contatiner it keeps the three of them 
overall<-dOverall(d_time,d_conv,d_peak_rate) #it uses geometric mean because everything we want acceptable

rsmOpt<-function(x,dObject){
  newdata<-data.frame(x1=x[1],x2=x[2],x3=x[3])
  pred_time<-predict(model_time, newdata)
  pred_peak<-predict(model_peak, newdata)
  pred_conv<-predict(model_conversion, newdata)
  #it takes the three of them
  out<-predict(dObject,
               data.frame(pred_time=pred_time,
                          pred_peak=pred_peak,
                          pred_conv=pred_conv))
  if (any(abs(x)>1)) out<-0  #the penalty and unconstrained ,if you are out of [-1,+1]^3 ->extrapolation 
  
  return(out)
  
}
#grid search Creates a data frame from all combinations of the supplied vectors or factors
searchGrid <- expand.grid(
  x1 = seq(-1, 1, length = 5),
  x2 = seq(-1, 1, length = 5),
  x3 = seq(-1, 1, length = 5)
)

#finds the optimum 
best <- NULL
for (i in 1:nrow(searchGrid)) {
  tmp <- optim(
    par     = as.numeric(searchGrid[i, ]),   
    fn      = rsmOpt,                          
    dObject = overall,                        
    control = list(fnscale = -1)               
  )
  if (is.null(best) || tmp$value > best$value) {
    best <- tmp
  }
}
#it keeps only one and then again only one ,it compares the previous with the new
#short circhit 
if (is.null(best) || tmp$value > best$value) {  
  best <- tmp
}

T_opt  <- 21  + best$par[1] * 6
S0_opt <- 265 + best$par[2] * 15
N0_opt <- 240 + best$par[3] * 100

cat("=== Optimal Fermentation Conditions ===\n")
cat(sprintf("  T  = %.2f °C\n", T_opt))
cat(sprintf("  S0 = %.2f g/L\n", S0_opt))
cat(sprintf("  N0 = %.2f mg/L\n", N0_opt))
cat(sprintf("  Overall D = %.4f\n", best$value))



#the model wanted to go further with the temperature but it cant because we have veto 


#

# 1. responses in the optimal  
opt_coded <- data.frame(x1 = best$par[1], x2 = best$par[2], x3 = best$par[3])
pred_time <- as.numeric(predict(model_time,       opt_coded))
pred_peak <- as.numeric(predict(model_peak,       opt_coded))
pred_conv <- as.numeric(predict(model_conversion, opt_coded))


optimal <- data.frame(
  T_C       = round(T_opt, 2),
  S0_gL     = round(S0_opt, 2),
  N0_mgL    = round(N0_opt, 2),
  x1_coded  = round(best$par[1], 4),
  x2_coded  = round(best$par[2], 4),
  x3_coded  = round(best$par[3], 4),
  pred_time = round(pred_time, 2),
  pred_peak = round(pred_peak, 4),
  pred_conv = round(pred_conv, 4),
  overall_D = round(best$value, 4)
)


write.csv(optimal, "results/tables/optimal_conditions.csv", row.names = FALSE)



