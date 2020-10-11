# https://www.business-science.io/business/2016/08/07/CustomerSegmentationPt1.html
library(readxl); library(tidyverse)
# Prep data
data = full_join(read_excel('orders.xlsx'),
                 read_excel('bikes.xlsx'),
                 by = c('product.id' = 'bike.id')) %>% 
  full_join(read_excel('bikeshops.xlsx'), by = c('customer.id' = 'bikeshop.id')) %>% 
  group_by(bikeshop.name, model, category1, category2, frame, price) %>% 
  summarize(total.qty = sum(quantity, na.rm = T)) %>% 
  spread(bikeshop.name, total.qty) %>%
  ungroup() %>%
  mutate(across(.cols = `Albuquerque Cycles`:`Wichita Speed`, 
                .fns = ~if_else(is.na(.x), 0, .x / sum(.x, na.rm = T))),
         price.cat = if_else(price >= median(price, na.rm = T), 'high', 'low'))

# K-means clustering
