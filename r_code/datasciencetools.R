# datasciencetools
# Matt O, 8/24/2016

################
## Dataframes ##
################

# basic functions for dataframes:

# removes rows with any NAs from a dataframe (returns df):
rm.na <- function(df){
  row.has.na <- apply(df, 1, function(x){any(is.na(x))})
  df[!row.has.na, ]
  }

###########
## Stats ##
###########
# https://www.r-bloggers.com/comparison-of-two-proportions-parametric-z-test-and-non-parametric-chi-squared-methods/
# this blog is wrong. these tests are the same.. prop.test is not chisqured.
# chisquared is done with chisq.test()

# comparing two proportions:
prop.test.z <- function(x1,x2,n1,n2){
     numerator = (x1/n1) - (x2/n2)
     p.common = (x1+x2) / (n1+n2)
     denominator = sqrt(p.common * (1-p.common) * (1/n1 + 1/n2))
     z.prop.ris = numerator / denominator
     return(z.prop.ris)
 }
# prop.test(x = c(30, 65), n = c(74, 103), correct = FALSE) # chi squared




# plotting

# from http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}






#df.ranksum(df, valcol, splitcol, ...)

#df.ranksum <- function(df, valcol, splitcol){
#
#  d <- rm.na(df[c(valcol,splitcol)])
#  falses
#  wilcox.test(as.numeric(x19$CMS_19[x19$hasNote==TRUE]),as.numeric(x19$CMS_19[x19$hasNote==FALSE]), correct=FALSE)
#}




























