# Reference R implementations on the inputs exported by pipelines/benchmark_r_reference.py.
#
#   Rscript benchmarks/r_reference.R <input_dir>
#
# Needs coloc, susieR, MendelianRandomization and jsonlite from CRAN. Writes r_results.json
# into <input_dir>.

suppressPackageStartupMessages({
  library(jsonlite)
  library(coloc)
  library(susieR)
  library(MendelianRandomization)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
meta <- fromJSON(file.path(dir, "meta.json"))
out <- list(
  versions = list(
    R = as.character(getRversion()),
    coloc = as.character(packageVersion("coloc")),
    susieR = as.character(packageVersion("susieR")),
    MendelianRandomization = as.character(packageVersion("MendelianRandomization"))
  ),
  loci = list(), mr = list()
)

for (locus in names(meta$loci)) {
  m <- meta$loci[[locus]]
  d <- read.csv(file.path(dir, paste0(locus, "_region.csv")))
  p <- nrow(d)
  R <- matrix(readBin(file.path(dir, paste0(locus, "_ld.bin")), "double", p * p, size = 8),
              p, p)
  ids <- paste0("v", seq_len(p) - 1)
  dimnames(R) <- list(ids, ids)
  z1 <- setNames(d$z1, ids)
  z2 <- setNames(d$z2, ids)

  t0 <- proc.time()[["elapsed"]]
  f1 <- susie_rss(z = z1, R = R, n = m$n1, L = 10)
  f2 <- susie_rss(z = z2, R = R, n = m$n2, L = 10)
  t_susie <- proc.time()[["elapsed"]] - t0

  cs <- function(f) {
    s <- f$sets$cs
    if (is.null(s)) return(list())
    lapply(unname(s), function(v) I(as.integer(v - 1)))
  }
  t0 <- proc.time()[["elapsed"]]
  cc <- coloc.susie(f1, f2, p12 = 5e-6)
  t_cs <- proc.time()[["elapsed"]] - t0
  summ <- if (is.null(cc$summary)) list() else lapply(seq_len(nrow(cc$summary)), function(i) {
    r <- cc$summary[i, ]
    list(PP.H4 = r$PP.H4.abf, PP.H3 = r$PP.H3.abf,
         hit1 = as.integer(sub("v", "", r$hit1)), hit2 = as.integer(sub("v", "", r$hit2)))
  })

  t0 <- proc.time()[["elapsed"]]
  abf <- coloc.abf(
    dataset1 = list(beta = setNames(d$beta1, ids), varbeta = setNames(d$se1^2, ids),
                    type = "quant", sdY = 1, N = m$n1, snp = ids),
    dataset2 = list(beta = setNames(d$beta2, ids), varbeta = setNames(d$se2^2, ids),
                    type = "cc", N = m$n2, snp = ids),
    p1 = 1e-4, p2 = 1e-4, p12 = 1e-5
  )
  t_abf <- proc.time()[["elapsed"]] - t0

  out$loci[[locus]] <- list(
    pip1 = as.numeric(susie_get_pip(f1)), pip2 = as.numeric(susie_get_pip(f2)),
    cs1 = cs(f1), cs2 = cs(f2),
    coloc_susie = summ,
    abf = as.list(abf$summary[c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")]),
    seconds = list(susie_two_traits = t_susie, coloc_susie = t_cs, coloc_abf = t_abf)
  )
}

h <- read.csv(file.path(dir, "mr_instruments.csv"))
t0 <- proc.time()[["elapsed"]]
for (tr in unique(h$trait)) {
  x <- h[h$trait == tr, ]
  inp <- mr_input(bx = x$beta_exp, bxse = x$se_exp, by = x$beta_out, byse = x$se_out)
  k <- nrow(x)
  iv <- mr_ivw(inp, model = if (k > 1) "random" else "fixed")
  r <- list(k = k, ivw = c(iv@Estimate, iv@StdError))
  if (k >= 3) {
    eg <- mr_egger(inp)
    wm <- mr_median(inp, weighting = "weighted", iterations = 1000, seed = 0)
    r$egger <- c(eg@Estimate, eg@StdError.Est, eg@Intercept)
    r$weighted_median <- c(wm@Estimate, wm@StdError)
  }
  out$mr[[tr]] <- r
}
out$mr_seconds <- proc.time()[["elapsed"]] - t0

write_json(out, file.path(dir, "r_results.json"), auto_unbox = TRUE, digits = NA)
