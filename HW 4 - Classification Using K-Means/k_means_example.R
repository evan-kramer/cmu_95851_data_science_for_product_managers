# https://www.business-science.io/business/2016/08/07/CustomerSegmentationPt1.html
library(readxl); library(tidyverse)
setwd("C:/Users/evan.kramer/Documents/CMU/Courses/2020-03/95851 - Making Products Count, Data Science for Product Managers/Assignments/HW 4 - Classification Using K-Means/")
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
data

ggplot(data, aes(x = category, y = category2, fill = frame)) + 
  geom_bar() +
  facet_grid(rows = frame)
