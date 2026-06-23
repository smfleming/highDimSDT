#######################################################################################
## 
## Simulations of a k-dimensional signal detection agent that must decide between
## stimuli s_1 and s_2, but computes confidence across all k dimensions.
##
## Questions? Contact Kevin O'Neill at kevin.o'neill@ucl.ac.uk
##
#######################################################################################
library(tidyverse)     # for data wrangling
library(ggdist)        # for distribution plots
library(LaplacesDemon) # for multivariate normal distributions
library(matrixStats)   # for efficient logSumExp() function
library(viridis)       # for color scales
library(patchwork)     # for multi-panel plots
library(emmeans)       # for linear regressions

#' softmax(x):
#'   calculate the softmax x
#' Arguments:
#'   x: a vector or matrix of values
#' Value:
#'   S: if x is a vector, the softmax of x.
#'      if x is a matrix, calculate the softmax for each row.
softmax <- function(x, beta=1) {
  if (is.vector(x)) {
    exp(beta*x - logSumExp(beta*x))        
  } else {
    exp(beta*x - rowLogSumExps(beta*x))
  }
}

#' Jsoftmax(x):
#'   calculate the Jacobian matrix of the softmax of x
#' Arguments:
#'   x: a vector of N values
#' Value:
#'   J: an NxN matrix where each entry J[i,j] represents
#'      the partial derivative of softmax(x)[i] with respect to x[j]
Jsoftmax <- function(x, beta=1) {
  s <- softmax(x, beta=beta)
  if (is.vector(x)) {
    beta * (diag(s) - outer(s, s))
  } else {
    apply(s, 1, function(x) beta * (diag(x) - outer(x, x)), simplify=FALSE)
  }
}

#######################################################################################
## calculate confidence for a range of X1 and X2,
## setting all other X's to zero
#######################################################################################
c <- expand_grid(mu_target=1,
                 x1=seq(0, 3, .01),
                 x2=seq(0, 3, .01),
                 k=c(2, 5, 10, 20)) |>
  mutate(X=pmap(list(x1, x2, k),
                function(x1, x2, k) c(x1, x2, rep(0, times=k-2))),
         posterior=map2(X, mu_target, softmax),
         a=map_dbl(posterior, ~ which.max(.[1:2])),
         confidence=map_dbl(posterior, ~ max(.[1:2])),
         dconf_chosen=pmap_dbl(list(X, a, mu_target),
                               function(x,a,m) Jsoftmax(x, m)[a,a]),
         dconf_unchosen=pmap_dbl(list(X, a, mu_target),
                                 function(x,a,m) Jsoftmax(x, m)[a,3-a]),
         peb=log(abs(dconf_chosen)) - log(abs(dconf_unchosen)))

#######################################################################################
## Fix X_2 = 1.5, X_{2:k} ~= 0, vary X_1
#######################################################################################
k <- 10
X_other <- rnorm(k-2, mean=0, sd=.05)  ##runif(k-2, 1, 1.2)
c2 <- expand_grid(mu_target=1,
                  x1=seq(0, 1.5, by=.01), x2=.75, k=k) |>
  mutate(X=map2(x1, x2, function(x1, x2) c(x1, x2, X_other)),
         posterior=map2(X, mu_target, softmax),
         a=map_dbl(posterior, ~ which.max(.[1:2])),
         confidence=map_dbl(posterior, ~ max(.[1:2])))

#######################################################################################
## Simulate confidence from signal detection model
#######################################################################################
##N=1000000
c3 <- expand_grid(k=c(2, 5, 10, 20), stimulus=1:2, mu_target=1, N=10000) |>
  group_by(mu_target, k, stimulus) |>
  mutate(mu=pmap(list(k, stimulus, mu_target),
                 function(k,s,m) if (s==1) c(m, rep(0, k-1)) else c(0, m, rep(0, k-2))),
         X=map2(N, mu, ~ rmvnpc(.x, mu=.y, U=diag(length(.y)))),
         posterior=map2(X, mu_target, softmax),
         choice=map(posterior, ~ apply(.[,1:2], 1, which.max)),
         confidence=map(posterior, ~ apply(.[,1:2], 1, max)),
         accuracy=map2(choice, stimulus, ~ as.integer(.x==.y)),
         X1=map(X, ~ .[,1]),
         X2=map(X, ~ .[,2]),
         X_chosen=map(X, ~ apply(.[,1:2], 1, max)),
         X_unchosen=map(X, ~ apply(.[,1:2], 1, min)),
         k=factor(k),
         mu_target=factor(mu_target))

c4 <- expand_grid(k=2:20, stimulus=1:2, mu_target=1, N=10000) |>
  group_by(k, stimulus) |>
  mutate(mu=pmap(list(k, stimulus, mu_target),
                 function(k,s,m) if (s==1) c(m, rep(0, k-1)) else c(0, m, rep(0, k-2))),
         X=map2(N, mu, ~ rmvnpc(.x, mu=.y, U=diag(length(.y)))),
         choice=map(X, ~ apply(.[,1:2], 1, which.max))) |>
  group_by(k, mu_target) |>
  mutate(peb=map2(X, choice, function(x,c)
    map2_dbl(Jsoftmax(x, beta=first(mu_target)), c,
             ~ log(abs(.x[.y,.y]))-log(abs(.x[.y,3-.y])))))

lms <- c3 |>
  group_by(mu_target) |>
  select(mu_target, k, accuracy, confidence, X_chosen, X_unchosen) |>
  unnest(c(accuracy, confidence, X_chosen, X_unchosen)) |>
  mutate(confidence=pmin(pmax(confidence, plogis(-35)), plogis(35))) |>  ## avoid under/overflow
  nest() |>
  mutate(m.accuracy=map(data, ~ glm(accuracy ~ k*(X_chosen + X_unchosen),
                                    family='binomial', data=.)),
         m.confidence=map(data, ~ lm(qlogis(confidence) ~ k*(X_chosen + X_unchosen),
                                     data=.)),
         betas=map2(m.accuracy, m.confidence,
                    ~ bind_rows(emmeans(.x, ~ X_chosen | k, at=list(X_chosen=0:1)) |>
                                  contrast(method='trt.vs.ctrl') |>
                                  as_tibble() |>
                                  mutate(dv='Accuracy', variable='X_chosen'),
                                emmeans(.x, ~ X_unchosen | k, at=list(X_unchosen=0:1)) |>
                                  contrast(method='trt.vs.ctrl') |>
                                  as_tibble() |>
                                  mutate(dv='Accuracy', variable='X_unchosen'),
                                emmeans(.y, ~ X_chosen | k, at=list(X_chosen=0:1)) |>
                                  contrast(method='trt.vs.ctrl') |>
                                  as_tibble() |>
                                  mutate(dv='Confidence', variable='X_chosen'),
                                emmeans(.y, ~ X_unchosen | k, at=list(X_unchosen=0:1)) |>
                                  contrast(method='trt.vs.ctrl') |>
                                  as_tibble() |>
                                  mutate(dv='Confidence', variable='X_unchosen')))) |>
  select(mu_target, betas) |>
  unnest(betas)



## plot confidence alongside its partial derivatives
for (m in unique(c$mu_target)) {
  peb.max <- c |> filter(mu_target==1) |> pull(peb) |> abs() |> max() |> round()
  p.k.confidence <- c |>
    filter(mu_target == m) |>
    ggplot(aes(x=x1, y=x2)) +
    geom_raster(aes(fill=confidence)) +
    geom_contour(aes(z=confidence), color='black', bins=6) +
    geom_hline(aes(yintercept=y), linetype='dashed', color='white',
               data=tibble(y=.75, k=10)) +
    scale_fill_viridis('Confidence', limits=c(0, 1))
  p.k.dconf <- c |>
    filter(mu_target == m) |>
    ggplot(aes(x=x1, y=x2)) +
    geom_raster(aes(fill=dconf_chosen)) +
    scale_fill_viridis('Change in\nConfidence\n(Chosen)',
                       option='magma', limits=c(0, NA))
  p.k.dconf2  <- c |>
    filter(mu_target == m) |>
    ggplot(aes(x=x1, y=x2)) +
    geom_raster(aes(fill=-dconf_unchosen)) +
    scale_fill_viridis('Change in\nConfidence\n(Unchosen)',
                       option='magma', limits=c(0, NA))
  p.k.peb <- c |>
    filter(mu_target == m) |>
    ggplot(aes(x=x1, y=x2)) +
    geom_raster(aes(fill=peb)) +
    scale_fill_distiller(name='PEB\n(log scale)', palette='PuOr', limits=c(-peb.max, peb.max))
  ## plot average PEB for simulated choices
  p.peb <- c4 |>
    filter(mu_target==m) |>
    unnest(peb) |>
    summarize(PEB=median(peb)) |>
    ggplot(aes(x=k, y=PEB)) +
    geom_line() +
    ylab('Average Positive Evidence Bias\n(log scale)') +
    scale_x_continuous('k', limits=c(0, NA)) +
    ##coord_fixed(ratio=5) +
    theme_light(18)
  ## plot out beta values separately by k
  p.beta <-lms |>
    filter(mu_target==m) |>
    ggplot(aes(x=variable, y=estimate, fill=k)) +
    geom_hline(yintercept=0) +
    geom_col(position=position_dodge(1)) +
    scale_fill_brewer(palette='Oranges') +
    ylab('Beta coefficients') +
    facet_wrap(~ dv, scales='free') +
    theme_classic(18) +
    theme(axis.title.x=element_blank())
  p.confidence.slice <- c2 |>
    filter(mu_target == m) |>
    mutate(dim=map(k, ~ seq(1, .))) |>
    unnest(c(dim, X, posterior)) |>
    mutate(dim=factor(dim)) |>
    mutate(dim_label=ifelse(dim==1, '1', ifelse(dim==2, '2', '3+'))) |>
    ggplot(aes(x=x1, y=posterior)) +
    geom_line(aes(y=confidence), linewidth=2, data=~ filter(., dim==1)) +
    geom_line(aes(color=dim_label, group=dim), linewidth=1) +
    scale_color_discrete('Stimulus') +
    xlab('X1') + ylab('Posterior Probability') +
    coord_cartesian(ylim=c(0, NA), expand=FALSE) +
    theme_classic(18)
  ## Figure 2
  (((p.k.confidence / p.k.dconf / p.k.dconf2 / p.k.peb) &
      facet_grid(~ k, labeller=label_both) &
      xlab('X1') & ylab('X2') &
      coord_fixed(expand=FALSE) &
      theme_classic(18)) |
     ((p.peb / free(p.beta) / p.confidence.slice) +
        plot_layout(heights=c(3, 2, 3)))) +
    plot_annotation(tag_levels='A') +
    plot_layout(widths=c(2, 1))
  ggsave(paste0('fig2_', m, '.pdf'), width=20, height=16)
}

## Plot simulated confidence by stimulus (as in Figure 3)
c3 |>
  unnest(c(X1, X2, confidence)) |>
  ggplot(aes(x=X1, y=X2, color=confidence)) +
  geom_point(size=.1) +
  facet_grid(stimulus ~ k, labeller=label_both) +
  scale_color_viridis('Confidence', limits=0:1) +
  coord_fixed(expand=FALSE) +
  theme_bw(18)
ggsave('variance.pdf', width=12, height=6)



#######################################################################################
## Determine effects on metacognitive sensitivity
#######################################################################################

## Plot posterior probabilities for target/nontarget stimulus
p.posterior <- c3 |>
  mutate(p.target=map2(posterior, stimulus, ~ .x[,.y]),
         p.nontarget=map2(posterior, stimulus, ~ .x[,3-.y])) |>
  unnest(c(p.target, p.nontarget)) |>
  select(k, stimulus, p.target, p.nontarget) |>
  pivot_longer(p.target:p.nontarget, names_prefix='p\\.') |>
  mutate(name=factor(name, levels=c('target', 'nontarget'),
                     labels=c('Target', 'Non-target'))) |>
  ggplot(aes(x=value, fill=name)) +
  stat_slab(alpha=.4, color=NA, adjust=2) +
  facet_grid(~ k, labeller=label_both) +
  scale_x_continuous('Posterior Probability', limits=c(0, 1),
                     labels=c('0', '.25', '.5', '.75', '1')) +
  scale_fill_manual('Dimension', values=c('darkgoldenrod3', 'grey70')) +
  theme_classic(18) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.line.y=element_blank())

## Plot confidence for correct/incorrect decisions
p.confidence.accuracy <- c3 |>
  unnest(c(confidence, accuracy)) |>
  mutate(accuracy=ifelse(accuracy, 'Correct', 'Incorrect')) |>
  ggplot(aes(x=confidence, fill=accuracy)) +
  stat_slab(alpha=.4, color=NA, adjust=2) +
  facet_grid(~ k, labeller=label_both) +
  scale_x_continuous('Confidence', limits=c(0, 1),
                     labels=c('0', '.25', '.5', '.75', '1')) +
  scale_fill_manual('Accuracy', values=c('darkgoldenrod3', 'grey70')) +
  theme_classic(18) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.line.y=element_blank())

## Plot Type 2 ROC by sliding a confidence criteria c across
## distributions of confidence for correct/incorrect decisions
p.roc2 <- c3 |>
  select(confidence, accuracy) |>
  unnest(c(confidence, accuracy)) |>
  mutate(accuracy=ifelse(accuracy, 'Correct', 'Incorrect')) |>
  group_by(k, accuracy) |>
  nest() |>
  pivot_wider(names_from=accuracy, values_from=data) |>
  expand_grid(c=seq(0, 1, by=.001)) |>
  mutate(hit2=map2_dbl(c, Correct, function(c,d) mean(d$confidence > c)),
         fa2=map2_dbl(c, Incorrect, function(c,d) mean(d$confidence > c))) |>
  ggplot(aes(x=fa2, y=hit2, color=factor(k))) +
  geom_abline(slope=1, intercept=0, linetype='dashed') +
  geom_line() +
  coord_fixed(xlim=c(0, 1), ylim=c(0, 1), expand=FALSE) +
  scale_color_viridis_d('k') +
  xlab('Type 2 False Alarm Rate') + ylab('Type 2 Hit Rate') +
  theme_classic(18) +
  theme(panel.grid.major=element_line())

## Figure 6
((p.posterior / p.confidence.accuracy) | p.roc2) +
  plot_layout(widths=c(2, 1)) +
  plot_annotation(tag_levels='A')
ggsave('metacognitive_sensitivity.pdf', width=16, height=6)




#######################################################################################
## Detection
#######################################################################################
#' compute P(present | X)
detection_posterior <- function(x, mu_target=1) {
  k <- length(x)
  ll <- c(log(k) + .5 * mu_target^2,
          logSumExp(x*mu_target))
  return(exp(ll - logSumExp(ll)))
}


## plotting function for drawing a circle
geom_circle <- function(center=c(0, 0), radius=1, n=101, ...) {
  d <- tibble(theta=seq(0, 2*pi, length.out=n),
              x=radius*cos(theta) + center[1],
              y=radius*sin(theta) + center[2])
  return(geom_path(aes(x=x, y=y), data=d, ...))
}

c <- expand_grid(mu_target=1,
                 x1=seq(-1.5, 3, .005),
                 x2=seq(-1.5, 3, .005)) |>
  mutate(X=pmap(list(x1, x2), function(x1, x2) c(x1, x2)),
         evidence=map_dbl(X, sum),
         posterior=map2(X, mu_target, ~ softmax(.x, .y)),
         a=map_dbl(posterior, ~ which.max(.[1:2])),
         confidence=map_dbl(posterior, ~ max(.[1:2])),
         choice=map_dbl(posterior, ~ qlogis(.[[2]])),
         detection_posterior=map(X, detection_posterior),
         detection_a=map_dbl(detection_posterior, which.max),
         detection_confidence=map_dbl(detection_posterior, max),
         detection_choice=map_dbl(detection_posterior, ~ qlogis(.[2])))


## Make detection vs discrimination plot
n.breaks <- 15
saturation <- .7
value <- 1
s1 <- 1/6
s2 <- 2/3
present <- 1/3
absent <- '#b3b3b3'
p.discrimination <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=choice)) +
  geom_hline(yintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_vline(xintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95), linewidth=1) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95), radius=.5, linewidth=1) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95), linewidth=1) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95), radius=.5, linewidth=1) +
  scale_fill_steps2(limits=c(-4.5, 4.5), breaks=seq(-4.5, 4.5, length.out=n.breaks),
                    labels=c('Certain\n"S1"', rep('', (n.breaks-3)/2), 'Guess',
                             rep('', (n.breaks-3)/2), 'Certain\n"S2"'),
                    low=hsv(s1, saturation, value), high=hsv(s2, saturation, value)) +
  ggtitle('Discrimination')
p.detection <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=detection_choice)) +
  geom_hline(yintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_vline(xintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95), linewidth=1) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95), radius=.5, linewidth=1) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95), linewidth=1) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95), radius=.5, linewidth=1) +
  geom_circle(color=absent, linewidth=1) +
  geom_circle(color=absent, linewidth=1, radius=.5) +
  scale_fill_steps2(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, length.out=n.breaks),
                    labels=c('Certain\n"Absent"', rep('', (n.breaks-3)/2), 'Guess',
                             rep('', (n.breaks-3)/2), 'Certain\n"Present"'),
                    low=absent, high=hsv(present, saturation, value)) +
  ggtitle('Detection')
((p.discrimination | p.detection) &
   coord_fixed(expand=FALSE) &
   xlab('X1') & ylab('X2') &
   theme_classic(18) &
   theme(legend.position='bottom',
         legend.title=element_blank())) +
  plot_annotation(tag_levels='A')






p.evidence <- ggplot() +
  geom_hline(yintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_vline(xintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95), linewidth=1, alpha=.15) +
  geom_circle(center=c(1, 0), color=hsv(s1, saturation, value*.95),
              radius=.5, linewidth=1, alpha=.15) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95), linewidth=1, alpha=.15) +
  geom_circle(center=c(0, 1), color=hsv(s2, saturation, value*.95),
              radius=.5, linewidth=1, alpha=.15) +
  geom_circle(color=absent, linewidth=1, alpha=.15) +
  geom_circle(color=absent, linewidth=1, radius=.5, alpha=.15) +
  geom_segment(aes(x=x, y=y, xend=xend, yend=yend),
               data=tibble(x=0.15, xend=2.65, y=2.65, yend=0.15),
               arrow=arrow(ends='both', type='closed')) +
  geom_segment(aes(x=x, y=y, xend=xend, yend=yend),
               data=tibble(x=c(0, 0), xend=c(1.75, 0),
                           y=c(0, 0), yend=c(0, 1.75)),
               arrow=arrow(type='closed')) +
  geom_text(aes(x=1.5, y=1.5), label='Balance of Evidence', color='black', angle=-45, size=3) +
  geom_text(aes(x=.75, y=-.33), color='black', size=3, lineheight=.9,
            label='Response\nCongruent\nEvidence') +
  geom_text(aes(x=-.33, y=.75), color='black', angle=-90, size=3, lineheight=.9,
            label='Response\nCongruent\nEvidence')

((p.evidence | p.discrimination | p.detection) &
   coord_fixed(xlim=c(-1.5, 3), ylim=c(-1.5, 3), expand=FALSE) &
   xlab('X1') & ylab('X2') &
   theme_classic(18) &
   theme(legend.title=element_blank())) +
  plot_annotation(tag_levels='A')
ggsave('fig1b.png', width=15, height=5)


p.discrimination2 <- ggplot(c, aes(x=x1, y=x2)) +
  geom_hline(yintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_vline(xintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_contour(aes(z=x1-x2), color='black', breaks=seq(-4.5, 4.5, length.out=n.breaks/2+1)) +
  geom_segment(aes(x=x, y=y, xend=xend, yend=yend, color=response),
               data=tibble(x=c(.75, .75), xend=c(-.75, 2.25),
                           y=c(.75, .75), yend=c(2.25, -.75),
                           response=c('s2', 's1')),
               linewidth=3, show.legend=FALSE, arrow=arrow(type='closed')) +
  scale_color_manual(values=c(hsv(s1, saturation, value*.95),
                              hsv(s2, saturation, value*.95))) +
  ggtitle('Balance of Evidence')
p.detection2 <- ggplot(c, aes(x=x1, y=x2)) +
  geom_hline(yintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_vline(xintercept=0, linetype='dashed', linewidth=.25, alpha=.5) +
  geom_contour(aes(z=pmax(x1, x2)), color='black', breaks=seq(-1.5, 3, length.out=n.breaks/2)) +
  geom_segment(aes(x=x, y=y, xend=xend, yend=yend, color=response),
               data=tibble(x=c(0, 0), xend=c(2.75, 0),
                           y=c(0, 0), yend=c(0, 2.75),
                           response=c('s1', 's2')),
               linewidth=3, arrow=arrow(type='closed'), show.legend=FALSE) +
  scale_color_manual(values=c(hsv(s1, saturation, value*.95),
                              hsv(s2, saturation, value*.95))) +
  ggtitle('Response Congruent Evidence')

(((p.discrimination | p.detection) / (p.discrimination2 | p.detection2)) &
   coord_fixed(expand=FALSE) &
   xlab('X1') & ylab('X2') &
   theme_classic(18) &
   theme(legend.position='bottom',
         legend.key.width=unit(2, 'cm'),
         legend.title=element_blank())) +
  plot_annotation(tag_levels='A')
ggsave('fig1.pdf', width=11, height=12)




## plot optimal confidence in detection tasks
c <- expand_grid(mu_target=1,
                 x1=seq(-6, 6, .05),
                 x2=seq(-6, 6, .05),
                 k=c(2, 5, 10, 20)) |>
  mutate(X=pmap(list(x1, x2, k),
                function(x1, x2, k) c(x1, x2, rep(0, times=k-2))),
         posterior=map2(X, mu_target, detection_posterior),
         posterior_absent=map_dbl(posterior, first),
         posterior_present=map_dbl(posterior, last),
         a=map_dbl(posterior, ~ which.max(.)-1),
         confidence=map_dbl(posterior, max))
ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=confidence)) +
  geom_contour(aes(z=confidence), color='black', bins=5) +
  scale_fill_viridis('Confidence', limits=c(.5, 1)) +
  facet_grid(~k, labeller=label_both) +
  coord_fixed(expand=FALSE) +
  xlab('X1') + ylab('X2') +
  theme_classic(18)
ggsave('detection.pdf', width=14, height=4)





#######################################################################################
## PEB flips with task demands
#######################################################################################
c <- expand_grid(mu_target=1,
                 x1=seq(0, 3, .1),
                 x2=seq(0, 3, .1),
                 k=c(2, 5, 10, 20)) |>
  mutate(X=pmap(list(x1, x2, k),
                function(x1, x2, k) c(x1, x2, rep(0, times=k-2))),
         posterior=map2(X, mu_target, softmax),
         chosen=map_dbl(posterior, ~ which.max(.[1:2])),
         a=map_dbl(posterior, ~ which.min(.[1:2])),
         confidence=map_dbl(posterior, ~ max(.[1:2])),
         dconf_chosen=pmap_dbl(list(X, a, mu_target), \(.x, .a, .m) Jsoftmax(.x, .m)[.a,3-.a]),
         dconf_unchosen=pmap_dbl(list(X, a, mu_target), \(.x, .a, .m) Jsoftmax(.x, .m)[.a,.a]),
         peb=map2_dbl(dconf_chosen, dconf_unchosen, ~ log(abs(.x)) - log(abs(.y))))


## plot confidence alongside its partial derivatives
p.k.confidence <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=confidence)) +
  geom_contour(aes(z=confidence), color='black') +
  geom_hline(aes(yintercept=y), linetype='dashed', color='white',
             data=tibble(y=.75, k=10)) +
  scale_fill_viridis('Confidence', limits=c(0, 1))

p.k.dconf <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=abs(dconf_chosen))) +
  scale_fill_viridis('Change in\nConfidence\n(Chosen)',
                     option='magma') #, limits=c(0, NA))
p.k.dconf2  <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=abs(dconf_unchosen))) +
  scale_fill_viridis('Change in\nConfidence\n(Unchosen)',
                     option='magma', limits=c(0, NA))
p.k.peb <- ggplot(c, aes(x=x1, y=x2)) +
  geom_raster(aes(fill=peb)) +
  scale_fill_distiller(name='PEB\n(log scale)', palette='PuOr', limits=c(-3, 3))


## Plot mean PEB as a function of dimensionality
p.peb <- c |> group_by(k) |>
  summarize(PEB=mean(peb)) |>
  ggplot(aes(x=k, y=PEB)) +
  geom_line() +
  ylab('Average Positive Evidence Bias\n(log scale)') +
  scale_x_continuous('k', limits=c(0, NA)) +
  coord_fixed(ratio=10) +
  theme_light(18)

(((p.k.confidence / p.k.dconf / p.k.dconf2 / p.k.peb) &
    facet_grid(~ k, labeller=label_both) &
    xlab('X1') & ylab('X2') &
    coord_fixed(expand=FALSE) &
    theme_classic(18)) | p.peb) +
  plot_annotation(tag_levels='A') +
  plot_layout(widths=c(2, 1))
ggsave('peb_flip.pdf', width=20, height=16)
