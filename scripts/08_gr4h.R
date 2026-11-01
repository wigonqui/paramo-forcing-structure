#!/usr/bin/env Rscript
# =============================================================================
# Stage 08 - GR4H calibration, validation and transfer
#
# Purpose
#   Calibrate the hourly conceptual model GR4H for every precipitation scenario of the site,
#   simulate the calibration and validation periods continuously, and transfer the
#   parameters without recalibration to the nested catchment.
#
# Methods
#   Michel calibration maximising KGE' (Kling et al., 2012) on hourly discharge in the
#   calibration catchment (gr4h$calibration_catchment). Warm-up uses all hours from
#   periods$warmup up to the start of the calibration, so the same code serves the direct and
#   the reversed split-sample experiments. Hours with missing precipitation in ANY scenario,
#   and the following gr4h$mask_window_h hours, are excluded from evaluation for every
#   scenario. Scenarios run in parallel (gr4h$workers). CemaNeige-GR4H is supported for
#   completeness (gr4h$models) but is not part of the final design.
#
# Tools
#   R >= 4.2, airGR (Coron et al., 2017), yaml, parallel (base R).
#
# Inputs
#   data/processed/<site>/forcing_<catchment>.csv (stage 07);
#   outputs/<site>/tables/03_catchments.csv, 07_hypsometry.csv
#
# Outputs
#   data/processed/<site>/simulations/GR4H__<scenario>__<catchment>.csv (Qobs, Qsim, states)
#   outputs/<site>/tables/08_parameters.csv, 08_metrics.csv
#
# Usage
#   PARAMO_SITE=chalpi Rscript scripts/08_gr4h.R
#   Rscript scripts/08_gr4h.R site=chalpi scenarios=gauge,fusion_corr
#   RStudio: define SITE <- "chalpi" (and optionally SCENARIOS) before sourcing the file
#
# Next stage
#   09 - LSTM benchmark.
# =============================================================================
suppressPackageStartupMessages({
  library(airGR)
  library(yaml)
  library(parallel)
})

# ----------------------------------------------------------------------------- project root
find_root <- function() {
  candidates <- Sys.getenv("PARAMO_ROOT")
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) candidates <- c(candidates, dirname(f))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- tryCatch(get("ofile", envir = sys.frame(i)), error = function(e) NULL)
    if (is.character(of)) candidates <- c(candidates, dirname(of))
  }
  p <- ""
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
  } else if (exists(".rs.api.getSourceEditorContext", mode = "function")) {
    p <- tryCatch(get(".rs.api.getSourceEditorContext")()$path, error = function(e) "")
  }
  if (length(p) == 1 && nzchar(p)) candidates <- c(candidates, dirname(p))
  candidates <- c(candidates, getwd())
  for (d in candidates[nzchar(candidates)]) {
    d <- normalizePath(d, winslash = "/", mustWork = FALSE)
    for (k in 0:3) {
      if (file.exists(file.path(d, "config.yaml"))) return(d)
      d <- dirname(d)
    }
  }
  stop("config.yaml not found. Run from the project folder or set ",
       "Sys.setenv(PARAMO_ROOT = 'path/to/project').")
}
ROOT <- find_root()

# ----------------------------------------------------------------------------- options
user_args <- if (interactive()) character(0) else commandArgs(trailingOnly = TRUE)
opt <- list()
for (a in user_args) {
  kv <- strsplit(a, "=")[[1]]
  if (length(kv) == 2) opt[[kv[1]]] <- strsplit(kv[2], ",")[[1]]
}
if (exists("MODELS", envir = globalenv())) opt$models <- get("MODELS", envir = globalenv())
if (exists("SCENARIOS", envir = globalenv())) opt$scenarios <- get("SCENARIOS", envir = globalenv())
if (exists("SITE", envir = globalenv())) opt$site <- get("SITE", envir = globalenv())

# ----------------------------------------------------------------------------- configuration
raw <- yaml::yaml.load(paste(readLines(file.path(ROOT, "config.yaml"), encoding = "UTF-8",
                                       warn = FALSE), collapse = "\n"))
SITE <- if (!is.null(opt$site)) opt$site[1] else Sys.getenv("PARAMO_SITE")
if (!nzchar(SITE)) SITE <- raw$project$default_site
if (is.null(raw$sites[[SITE]])) stop("Site not defined in config.yaml: ", SITE)
site_cfg <- raw$sites[[SITE]]
experiments <- site_cfg$experiments
site_cfg$experiments <- NULL
cfg <- modifyList(raw[setdiff(names(raw), "sites")], site_cfg)
cfg$paths <- lapply(cfg$paths, function(p) gsub("{site}", SITE, p, fixed = TRUE))
EXPERIMENT <- Sys.getenv("PARAMO_EXPERIMENT")
if (nzchar(EXPERIMENT)) {
  if (is.null(experiments[[EXPERIMENT]])) stop("Experiment not defined for ", SITE, ": ", EXPERIMENT)
  cfg <- modifyList(cfg, experiments[[EXPERIMENT]])
  for (k in c("processed", "tables", "figures")) {
    cfg$paths[[k]] <- paste0(cfg$paths[[k]], "_", EXPERIMENT)
  }
}
dir_processed <- file.path(ROOT, cfg$paths$processed)
dir_tables <- file.path(ROOT, cfg$paths$tables)
dir_sim <- file.path(dir_processed, "simulations")
dir.create(dir_sim, showWarnings = FALSE, recursive = TRUE)
dir.create(dir_tables, showWarnings = FALSE, recursive = TRUE)

say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "| 08_gr4h |", sprintf(...), "\n")
say("Project: %s | site: %s%s", ROOT, SITE,
    if (nzchar(EXPERIMENT)) paste0(" | experiment: ", EXPERIMENT) else "")
gr <- cfg$gr4h
models <- if (!is.null(opt$models)) opt$models else unlist(gr$models)

# ----------------------------------------------------------------------------- data
read_forcing <- function(catchment) {
  df <- read.csv(file.path(dir_processed, sprintf("forcing_%s.csv", catchment)),
                 check.names = FALSE)
  df$date <- as.POSIXct(df$time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
  midnight <- is.na(df$date)
  df$date[midnight] <- as.POSIXct(df$time[midnight], tz = "UTC", format = "%Y-%m-%d")
  df
}
day_start <- function(d) as.POSIXct(d, tz = "UTC")
day_end <- function(d) as.POSIXct(paste(d, "23:00:00"), tz = "UTC")
span <- function(F, a, b) which(F$date >= a & F$date <= b)

gap_mask <- function(df, scenarios, window) {
  missing <- Reduce(`|`, lapply(scenarios, function(s) !is.finite(df[[paste0("P__", s)]])))
  as.logical(stats::filter(as.numeric(missing), rep(1, window + 1), sides = 1) > 0) |
    is.na(missing)
}

CC <- gr$calibration_catchment
CT <- gr$transfer_catchment
Fc <- read_forcing(CC)
Ft <- read_forcing(CT)
attrs <- read.csv(file.path(dir_tables, "03_catchments.csv"))
hypso <- read.csv(file.path(dir_tables, "07_hypsometry.csv"))
all_scenarios <- sub("^P__", "", grep("^P__", names(Fc), value = TRUE))
scenarios <- if (!is.null(opt$scenarios)) opt$scenarios else all_scenarios
mask_c <- gap_mask(Fc, all_scenarios, gr$mask_window_h)
mask_t <- gap_mask(Ft, all_scenarios, gr$mask_window_h)

per <- cfg$periods
sim_start <- min(day_start(per$calibration[[1]]), day_start(per$validation[[1]]))
sim_end <- max(day_end(per$calibration[[2]]), day_end(per$validation[[2]]),
               day_end(per$transfer[[2]]))
warm_start <- day_start(per$warmup[[1]])
idx <- list(
  cal = span(Fc, day_start(per$calibration[[1]]), day_end(per$calibration[[2]])),
  wu_cal = span(Fc, warm_start, day_start(per$calibration[[1]]) - 3600),
  wu_c = span(Fc, warm_start, sim_start - 3600), sim_c = span(Fc, sim_start, sim_end),
  wu_t = span(Ft, warm_start, sim_start - 3600), sim_t = span(Ft, sim_start, sim_end))

# ----------------------------------------------------------------------------- one scenario
run_scenario <- function(task) {
  suppressPackageStartupMessages(library(airGR))
  model <- task$model
  scen <- task$scenario
  FUN <- list(GR4H = RunModel_GR4H, CemaNeigeGR4H = RunModel_CemaNeigeGR4H)[[model]]

  metrics <- function(o, s) {
    ok <- is.finite(o) & is.finite(s)
    o <- o[ok]; s <- s[ok]
    if (length(o) < 100) return(c(kge = NA, r = NA, beta = NA, gamma = NA, nse = NA,
                                  nse_log = NA, kge_sqrt = NA, n = length(o)))
    k <- function(o, s) {
      r <- cor(o, s); b <- mean(s) / mean(o); g <- (sd(s) / mean(s)) / (sd(o) / mean(o))
      c(1 - sqrt((r - 1)^2 + (b - 1)^2 + (g - 1)^2), r, b, g)
    }
    kk <- k(o, s); eps <- mean(o) / 100
    c(kge = kk[1], r = kk[2], beta = kk[3], gamma = kk[4],
      nse = 1 - sum((s - o)^2) / sum((o - mean(o))^2),
      nse_log = 1 - sum((log(s + eps) - log(o + eps))^2) /
        sum((log(o + eps) - mean(log(o + eps)))^2),
      kge_sqrt = k(sqrt(o), sqrt(s))[1], n = length(o))
  }
  inputs <- function(F, catchment) {
    P <- F[[paste0("P__", scen)]]; P[!is.finite(P) | P < 0] <- 0
    E <- F$PET; E[!is.finite(E) | E < 0] <- 0
    if (model == "GR4H") {
      return(CreateInputsModel(FUN, DatesR = F$date, Precip = P, PotEvap = E, verbose = FALSE))
    }
    Tm <- F$T_catchment
    if (any(!is.finite(Tm))) {
      ok <- is.finite(Tm)
      Tm <- approx(which(ok), Tm[ok], xout = seq_along(Tm), rule = 2)$y
    }
    CreateInputsModel(FUN, DatesR = F$date, Precip = P, PotEvap = E, TempMean = Tm,
                      ZInputs = attrs$z_mean[attrs$catchment == catchment],
                      HypsoData = hypso[[catchment]], NLayers = 5, verbose = FALSE)
  }
  run_opt <- function(inp, wu, run) suppressWarnings(CreateRunOptions(
    FUN, InputsModel = inp, IndPeriod_WarmUp = wu, IndPeriod_Run = run, warnings = FALSE,
    verbose = FALSE))
  states <- function(out) {
    e <- data.frame(Qsim = out$Qsim, Prod = out$Prod, Rout = out$Rout, AE = out$AE)
    if (model == "CemaNeigeGR4H") {
      e$SnowPack <- Reduce(`+`, lapply(out$CemaNeigeLayers, function(l) l$SnowPack)) /
        length(out$CemaNeigeLayers)
    }
    e
  }
  t0 <- Sys.time()
  inp_c <- inputs(Fc, CC)
  opt_cal <- run_opt(inp_c, idx$wu_cal, idx$cal)
  q_cal <- Fc$Qobs_mm[idx$cal]
  crit <- CreateInputsCrit(ErrorCrit_KGE2, InputsModel = inp_c, RunOptions = opt_cal,
                           Obs = q_cal, BoolCrit = !mask_c[idx$cal] & !is.na(q_cal),
                           warnings = FALSE)
  cal <- Calibration_Michel(InputsModel = inp_c, RunOptions = opt_cal, InputsCrit = crit,
                            CalibOptions = CreateCalibOptions(FUN), FUN_MOD = FUN,
                            verbose = FALSE)
  par <- cal$ParamFinalR
  save_sim <- function(sim, F, ix, mask, catchment) {
    out <- cbind(time = format(F$date[ix], "%Y-%m-%d %H:%M:%S"), Qobs = F$Qobs_mm[ix], sim,
                 excluded = mask[ix])
    write.csv(out, file.path(dir_sim, sprintf("%s__%s__%s.csv", model, scen, catchment)),
              row.names = FALSE)
    out
  }
  sc <- save_sim(states(FUN(InputsModel = inp_c, RunOptions = run_opt(inp_c, idx$wu_c, idx$sim_c),
                            Param = par)), Fc, idx$sim_c, mask_c, CC)
  inp_t <- inputs(Ft, CT)
  st <- save_sim(states(FUN(InputsModel = inp_t, RunOptions = run_opt(inp_t, idx$wu_t, idx$sim_t),
                            Param = par)), Ft, idx$sim_t, mask_t, CT)
  rows <- list()
  for (ev in list(list(CC, "calibration", sc, per$calibration),
                  list(CC, "validation", sc, per$validation),
                  list(CT, "transfer", st, per$transfer))) {
    s <- ev[[3]]
    f <- as.POSIXct(s$time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
    sel <- f >= day_start(ev[[4]][[1]]) & f <= day_end(ev[[4]][[2]]) & !s$excluded
    o <- s$Qobs[sel]; m <- s$Qsim[sel]
    day <- format(f[sel], "%Y-%m-%d")
    nh <- tapply(!is.na(o), day, sum)
    od <- tapply(o, day, mean, na.rm = TRUE)[nh >= 20]
    md <- tapply(m, day, mean)[nh >= 20]
    for (scale in c("hourly", "daily")) {
      mm <- if (scale == "hourly") metrics(o, m) else metrics(od, md)
      rows[[length(rows) + 1]] <- data.frame(model = model, scenario = scen,
                                             catchment = ev[[1]], period = ev[[2]],
                                             scale = scale, t(round(mm, 4)))
    }
  }
  list(params = data.frame(model = model, scenario = scen, catchment = CC,
                           parameter = paste0("X", seq_along(par)), value = par,
                           kge_calibration = cal$CritFinal),
       metrics = do.call(rbind, rows),
       seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
}

# ----------------------------------------------------------------------------- run
tasks <- lapply(as.vector(outer(models, scenarios, paste, sep = "|")), function(x) {
  p <- strsplit(x, "|", fixed = TRUE)[[1]]
  list(model = p[1], scenario = p[2])
})
workers <- max(1, min(as.integer(gr$workers), length(tasks), parallel::detectCores()))
say("Models: %s | scenarios: %d | workers: %d", paste(models, collapse = ", "),
    length(scenarios), workers)
if (workers > 1) {
  cl <- makeCluster(workers)
  on.exit(stopCluster(cl), add = TRUE)
  clusterExport(cl, c("Fc", "Ft", "CC", "CT", "attrs", "hypso", "mask_c", "mask_t", "idx",
                      "per", "dir_sim", "day_start", "day_end"))
  results <- parLapply(cl, tasks, run_scenario)
} else {
  results <- lapply(tasks, run_scenario)
}
for (r in results) {
  h <- r$metrics[r$metrics$scale == "hourly", ]
  say("%-8s %-16s KGE' hourly: calibration=%.3f | validation=%.3f | %s=%.3f (%.0f s)",
      h$model[1], h$scenario[1], h$kge[h$period == "calibration"],
      h$kge[h$period == "validation"], CT, h$kge[h$period == "transfer"], r$seconds)
}
write.csv(do.call(rbind, lapply(results, `[[`, "params")),
          file.path(dir_tables, "08_parameters.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(results, `[[`, "metrics")),
          file.path(dir_tables, "08_metrics.csv"), row.names = FALSE)
say("Finished: %d runs", length(results))
