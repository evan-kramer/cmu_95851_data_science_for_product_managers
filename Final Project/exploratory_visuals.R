# Exploratory Visuals
# Evan Kramer

# Attach packages
library(tidyverse); library(lubridate); library(readxl)
setwd("C:/Users/evan.kramer/Documents/CMU/Courses/2020-03/95851 - Making Products Count, Data Science for Product Managers/Assignments/Final Project")

# Load data
data = map(
  .x = list.files("Deloitte_Digital_Democracy_data/") %>% 
    .[str_detect(., ".xlsx")],
  .f = ~read_excel(str_c("Deloitte_Digital_Democracy_data/", .x))
)
data[[4]] = select(data[[4]], varname:file)
# Replace original names
str_replace_all(
  names(data[[1]]), 
  names(data[[1]]), 
  data[[4]]$varname_short[data[[4]]$file == "DDS9_Data_Extract_with_labels"]
)

anti_join(
  tibble(names = names(data[[1]])),
  data[[4]],
  # by = c("varname" = "names")
  by = c("names" = "varname")
)
