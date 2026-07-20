# R package dependencies for the Wine Innovation Engine.
# Run once to set up the R environment:  Rscript r/install_packages.R

required <- c(
  "rsm",           # response surface methodology (Box-Behnken fitting)
  "desirability",  # Derringer-Suich desirability optimisation
  "rprojroot",     # resolve repo-relative paths so the Rmd knits anywhere
  "rmarkdown",     # knit rsm_fit.Rmd
  "knitr",         # chunk engine used by the Rmd
  "testthat"       # unit testing
)

installed <- rownames(installed.packages())
to_install <- required[!(required %in% installed)]

if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  message("All required R packages already installed.")
}
