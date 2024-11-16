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
    return(beta * (diag(s) - outer(s, s)))
}

#######################################################################################
## calculate confidence for a range of X1 and X2,
## setting all other X's to zero
#######################################################################################
c <- expand_grid(x1=seq(0, 3, .01),
                 x2=seq(0, 3, .01),
                 k=c(2, 5, 10, 20)) %>%
    mutate(X=pmap(list(x1, x2, k),
                  function(x1, x2, k) c(x1, x2, rep(0, times=k-2))),
           posterior=map(X, softmax),
           a=map_dbl(posterior, ~ which.max(.[1:2])),
           confidence=map_dbl(posterior, ~ max(.[1:2])),
           dconf_chosen=map2_dbl(X, a, ~ Jsoftmax(.x)[.y,.y]),
           dconf_unchosen=map2_dbl(X, a, ~ Jsoftmax(.x)[.y,3-.y]),
           peb=map2_dbl(X, a, ~ logSumExp(.x[-.y] - .x[3-.y])))

## plot confidence alongside its partial derivatives
p.k.confidence <- ggplot(c, aes(x=x1, y=x2)) +
    geom_raster(aes(fill=confidence)) +
    geom_contour(aes(z=confidence), color='black') +
    geom_hline(aes(yintercept=y), linetype='dashed', color='white',
               data=tibble(y=.75, k=10)) +
    scale_fill_viridis('Confidence', limits=c(0, 1))
p.k.dconf <- ggplot(c, aes(x=x1, y=x2)) +
    geom_raster(aes(fill=dconf_chosen)) +
    scale_fill_viridis('Change in\nConfidence\n(Chosen)',
                       option='magma', limits=c(0, .26))
p.k.dconf2  <- ggplot(c, aes(x=x1, y=x2)) +
    geom_raster(aes(fill=-dconf_unchosen)) +
    scale_fill_viridis('Change in\nConfidence\n(Unchosen)',
                       option='magma', limits=c(0, .26))
p.k.peb <- ggplot(c, aes(x=x1, y=x2)) +
    geom_raster(aes(fill=peb)) +
    scale_fill_distiller(name='PEB\n(log scale)', palette='PuOr', limits=c(-3, 3))

## Plot mean PEB as a function of dimensionality
p.peb <- c %>% group_by(k) %>%
    summarize(PEB=mean(peb)) %>%
    ggplot(aes(x=k, y=PEB)) +
    geom_line() +
    ylab('Average Positive Evidence Bias\n(log scale)') +
    scale_x_continuous('k', limits=c(0, NA)) +
    coord_fixed(ratio=10) +
    theme_light(18)

#######################################################################################
## Fix X_2 = 1.5, X_{2:k} ~= 0, vary X_1
#######################################################################################
k <- 10
X_other <- rnorm(k-2, mean=0, sd=.05)  ##runif(k-2, 1, 1.2)
c2 <- expand_grid(x1=seq(0, 1.5, by=.01), x2=.75, k=k) %>%
    mutate(X=map2(x1, x2, function(x1, x2) c(x1, x2, X_other)),
           posterior=map(X, softmax),
           a=map_dbl(posterior, ~ which.max(.[1:2])),
           confidence=map_dbl(posterior, ~ max(.[1:2])))

p.confidence.slice <- c2 %>%
    mutate(dim=map(k, ~ seq(1, .))) %>%
    unnest(c(dim, X, posterior)) %>%
    mutate(dim=factor(dim)) %>%
    mutate(dim_label=ifelse(dim==1, '1', ifelse(dim==2, '2', '3+'))) %>%
    ggplot(aes(x=x1, y=posterior)) +
    geom_line(aes(y=confidence), linewidth=2, data=~ filter(., dim==1)) +
    geom_line(aes(color=dim_label, group=dim), linewidth=1) +
    scale_color_discrete('Stimulus') +
    xlab('X1') + ylab('Posterior Probability') +
    coord_cartesian(ylim=c(0, NA), expand=FALSE) +
    theme_classic(18)

#######################################################################################
## Simulate confidence from signal detection model
#######################################################################################
c3 <- expand_grid(k=c(2, 5, 10, 20), stimulus=1:2, N=50000) %>%
    group_by(k, stimulus) %>%
    mutate(mu=map2(k, stimulus, ~ if (.y==1) c(1, rep(0, .x-1)) else c(0, 1, rep(0, .x-2))),
           X=map2(N, mu, ~ rmvnpc(.x, mu=.y, U=diag(length(.y)))),
           posterior=map(X, softmax),
           choice=map(posterior, ~ apply(.[,1:2], 1, which.max)),
           confidence=map(posterior, ~ apply(.[,1:2], 1, max)),
           accuracy=map2(choice, stimulus, ~ as.integer(.x==.y)),
           X1=map(X, ~ .[,1]),
           X2=map(X, ~ .[,2]),
           X_chosen=map(X, ~ apply(.[,1:2], 1, max)),
           X_unchosen=map(X, ~ apply(.[,1:2], 1, min))) %>%
    mutate(k=factor(k))

## Run a linear regression on accuracy/confidence
m.accuracy <- glm(accuracy ~ k*X_chosen + k*X_unchosen, family='binomial',
                  data=unnest(c3, c(accuracy, X_chosen, X_unchosen)))
summary(m.accuracy)
m.confidence <- lm(confidence ~ k*X_chosen + k*X_unchosen,
                   data=unnest(c3, c(confidence, X_chosen, X_unchosen)))
summary(m.confidence)

## extract coefficients of chosen/unchosen evidence separa
betas <- bind_rows(emmeans(m.accuracy, ~ X_chosen | k, at=list(X_chosen=0:1)) %>%
                   contrast(method='trt.vs.ctrl') %>%
                   as_tibble %>%
                   mutate(dv='Accuracy', variable='X_chosen'),
                   emmeans(m.accuracy, ~ X_unchosen | k, at=list(X_unchosen=0:1)) %>%
                   contrast(method='trt.vs.ctrl') %>%
                   as_tibble %>%
                   mutate(dv='Accuracy', variable='X_unchosen'),
                   emmeans(m.confidence, ~ X_chosen | k, at=list(X_chosen=0:1)) %>%
                   contrast(method='trt.vs.ctrl') %>%
                   as_tibble %>%
                   mutate(dv='Confidence', variable='X_chosen'),
                   emmeans(m.confidence, ~ X_unchosen | k, at=list(X_unchosen=0:1)) %>%
                   contrast(method='trt.vs.ctrl') %>%
                   as_tibble %>%
                   mutate(dv='Confidence', variable='X_unchosen'))

## plot out beta values separately by k
p.beta <- ggplot(betas, aes(x=variable, y=estimate, fill=k)) +
    geom_hline(yintercept=0) +
    geom_col(position=position_dodge(1)) +
    scale_fill_brewer(palette='Oranges') +
    ylab('Beta coefficients') +
    facet_wrap(~ dv, scales='free') +
    theme_classic(18) +
    theme(axis.title.x=element_blank())

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
ggsave('fig2.pdf', width=20, height=16)

## Plot simulated confidence by stimulus (as in Figure 3)
c3 %>%
    unnest(c(X1, X2, confidence)) %>%
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
p.posterior <- c3 %>%
    mutate(p.target=map2(posterior, stimulus, ~ .x[,.y]),
           p.nontarget=map2(posterior, stimulus, ~ .x[,3-.y])) %>%
    unnest(c(p.target, p.nontarget)) %>%
    select(k, stimulus, p.target, p.nontarget) %>%
    pivot_longer(p.target:p.nontarget, names_prefix='p\\.') %>%
    mutate(name=factor(name, levels=c('target', 'nontarget'),
                       labels=c('Target', 'Non-target'))) %>%
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
p.confidence.accuracy <- c3 %>%
    unnest(c(confidence, accuracy)) %>%
    mutate(accuracy=ifelse(accuracy, 'Correct', 'Incorrect')) %>%
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
p.roc2 <- c3 %>%
    select(confidence, accuracy) %>%
    unnest(c(confidence, accuracy)) %>%
    mutate(accuracy=ifelse(accuracy, 'Correct', 'Incorrect')) %>%
    group_by(k, accuracy) %>%
    nest() %>%
    pivot_wider(names_from=accuracy, values_from=data) %>%
    expand_grid(c=seq(0, 1, by=.001)) %>%
    mutate(hit2=map2_dbl(c, Correct, function(c,d) mean(d$confidence > c)),
           fa2=map2_dbl(c, Incorrect, function(c,d) mean(d$confidence > c))) %>%
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


