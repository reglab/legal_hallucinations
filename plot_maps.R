## Libs
set.seed(47)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(stringr)
library(ggplot2)
library(usmap)
library(viridis)
library(Hmisc)
library(sf)
library(reticulate)
settings <- import('settings')

# Load fonts
library(extrafont)
font_family <- 'sans'
tryCatch({
  font_import(pattern = 'Times New Roman', prompt = FALSE)
  font_family <- 'Times New Roman'
})

# File path
BASE_DIRECTORY = settings$BASE_DIRECTORY

# Models
apis <- c('OpenAIChatGpt4', 'OpenAIChat', 'GooglePaLMCompletion', 'LlamaChat')

# Filter to >= 1981 flag
# Set to FALSE to replicate supplemental map in appendix
FILTER_1981 <- TRUE

##############
# USCOA map
##############

# COA tasks
tasks <- c(
  'case_existence_few_shot',
  'court_id_few_shot',
  'citation_retrieval_few_shot',
  'majority_author_few_shot',
  'affirm_reverse_few_shot',
  'quotation_few_shot',
  'cited_precedent_few_shot'
)

# Pool results
results_pooled <- data.frame()
for (task in tasks) {
  for (api in apis) {
    # Load data
    setwd(str_glue('{BASE_DIRECTORY}/results/tasks/coa'))
    results_files <- list.files(pattern = str_glue('{task}_{api}_results_temp='))
    result <- read.csv(tail(results_files, n = 1))
    stopifnot(nrow(result) == 5000)

    # Recode abstentions as correct responses
    result <- result %>% mutate(correctness = as.integer(if_else(correctness == -99, 100, correctness)))

    # Filter out pre-1981 cases (establishment of 11th Circuit) if requested
    if (FILTER_1981) {
      result <- result %>% filter(year >= 1981)
    }

    # Set hallucination cutoff
    result$hallucination <- ifelse(result$correctness <= 72, 1, 0)

    # Pool results
    result <- result %>% select(court, hallucination)
    results_pooled <- rbind(results_pooled, result)
  }
}

# Prepare data for figure
results_by_circuit <- results_pooled %>% group_by(court) %>%
  summarise(hallucination_rate = mean(hallucination),
            count = n(),
            ci_lower = binconf(x = sum(hallucination), n = n(), alpha = 0.05)[2],
            ci_upper = binconf(x = sum(hallucination), n = n(), alpha = 0.05)[3],
            .groups = 'drop')

results_by_circuit$NAME <- ''
results_by_circuit$NAME[results_by_circuit$court == 1] <- 'FIRST CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 2] <- 'SECOND CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 3] <- 'THIRD CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 4] <- 'FOURTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 5] <- 'FIFTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 6] <- 'SIXTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 7] <- 'SEVENTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 8] <- 'EIGHTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 9] <- 'NINTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 10] <- 'TENTH CIRCUIT'
results_by_circuit$NAME[results_by_circuit$court == 11] <- 'ELEVENTH CIRCUIT'

courts_shape <- st_read(str_glue('{BASE_DIRECTORY}/data/shapefiles/US_CourtOfAppealsCircuits.shp'))
merged_data <- merge(courts_shape, results_by_circuit, by = 'NAME')

# Create main map
p <- ggplot(data = merged_data) +
  geom_sf(aes(fill = hallucination_rate), color = 'white') +
  coord_sf(xlim = c(-125, -66), ylim = c(24, 50), expand = FALSE) +
  scale_fill_viridis_c(direction = -1, limits = c(floor(min(results_by_circuit$hallucination_rate) * 1e5) / 1e5, ceiling(max(results_by_circuit$hallucination_rate) * 1e2) / 1e2)) +
  theme_void() +
  labs(fill = 'Hallucination\nRate') +
  theme(legend.position = 'right',
        legend.key.height = unit(0.9, 'cm'),
        text = element_text(size = 24, family = font_family),
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank())

# Create DC polygon to add
# Uncomment below to re-generate DC polygon from scratch; otherwise, just load
# the pre-calculated polygon from disk
#
# dc_data <- usmapdata::us_map() %>% filter(abbr == 'DC')
# dc_sf <- st_as_sf(dc_data, coords = c('x', 'y'))
# dc_sf <- st_set_crs(dc_sf, usmap_crs())
# dc_sf <- st_transform(dc_sf, crs = 'EPSG:4326')
# scale_and_shift <- function (geometry, scale_factor, x_shift, y_shift) {
#   coords <- matrix(st_coordinates(geometry), ncol = 2)
#   scaled_coords <- coords * scale_factor
#   translation_matrix <- matrix(rep(c(x_shift, y_shift), times = nrow(coords)), ncol = 2, byrow = TRUE)
#   final_coords <- scaled_coords + translation_matrix
#   new_geometry <- st_polygon(list(final_coords))
#   new_geometry <- st_sfc(new_geometry, crs = 'EPSG:4326')
#   return(new_geometry)
# }
# dc_transformed <- scale_and_shift(dc_sf, 35, 2608, -1337)

dc_transformed <- st_read(str_glue('{BASE_DIRECTORY}/data/shapefiles/dc_shapefile.shp'))

# Calculate colors
colors_mapped <- viridis(length(results_by_circuit$hallucination_rate), option = 'D', direction = -1)[rank(results_by_circuit$hallucination_rate)]

# Add DC and Fed. Circ. polygons
p <- p +
  geom_sf(data = dc_transformed, color = 'black', fill = colors_mapped[12], linewidth = 0.5) +
  annotate('text', x = -94, y = 23, label = 'D.C. Cir.', color = 'black', family = font_family, size = 7) +
  geom_rect(aes(xmin = -111, xmax = -103, ymin = 22, ymax = 26), fill = colors_mapped[13], color = 'black', linewidth = 0.5) +
  annotate('text', x = -116, y = 23, label = 'Fed. Cir.', color = 'black', family = font_family, size = 7) +
  coord_sf(xlim = c(-125, -66), ylim = c(20, 50), expand = FALSE)

# Save
if (FILTER_1981) {
  save_path <- str_glue('{BASE_DIRECTORY}/results/figures/pdf/hallucination_by_coa_geography_since_1981.pdf')
} else {
  save_path <- str_glue('{BASE_DIRECTORY}/results/figures/pdf/hallucination_by_coa_geography.pdf')
}
ggsave(filename = save_path, plot = p, device = 'pdf', width = 9, height = 7)

# Test difference between Second Circuit and Ninth Circuit
# prop.test(
#   c(results_by_circuit$hallucination_rate[[2]] * results_by_circuit$count[[2]], results_by_circuit$hallucination_rate[[9]] * results_by_circuit$count[[9]]),
#   c(results_by_circuit$count[[2]], results_by_circuit$count[[9]])
# )

##############
# USDC map
##############

# USDC tasks
tasks <- c(
  'case_existence_few_shot',
  'court_id_few_shot',
  'citation_retrieval_few_shot',
  'majority_author_few_shot',
  'quotation_few_shot',
  'cited_precedent_few_shot'
)

# Pool results
results_pooled <- data.frame()
for (task in tasks) {
  for (api in apis) {
    # Load data
    setwd(str_glue('{BASE_DIRECTORY}/results/tasks/usdc'))
    results_files <- list.files(pattern = str_glue('{task}_{api}_results_temp='))
    result <- read.csv(tail(results_files, n = 1))
    stopifnot(nrow(result) == 5000)

    # Recode abstentions as correct responses
    result <- result %>% mutate(correctness = as.integer(if_else(correctness == -99, 100, correctness)))

    # Filter out non-state
    result <- result %>% filter(state != 'misc') %>% filter(state != 'Puerto Rico') %>% filter(state != 'Virgin Islands')

    # Set hallucination cutoff
    result$hallucination <- ifelse(result$correctness <= 72, 1, 0)

    # Pool results
    result <- result %>% select(state, hallucination)
    results_pooled <- rbind(results_pooled, result)
  }
}

# Create figure
results_by_state <- results_pooled %>% group_by(state) %>%
  summarise(hallucination_rate = mean(hallucination),
            count = n(),
            ci_lower = binconf(x = sum(hallucination), n = n(), alpha = 0.05)[2],
            ci_upper = binconf(x = sum(hallucination), n = n(), alpha = 0.05)[3],
            .groups = 'drop')

color_limit_low <- floor(min(results_by_state$hallucination_rate) * 1e5) / 1e5
color_limit_high <- ceiling(max(results_by_state$hallucination_rate) * 1e2) / 1e2

p2 <- plot_usmap('states', data = results_by_state, values = 'hallucination_rate') +
  scale_fill_viridis_c(direction = -1, limits = c(color_limit_low, color_limit_high), breaks = seq(from = round(color_limit_low, 2), to = color_limit_high, by = 0.02)) +
  theme_void() +
  labs(fill = 'Hallucination\nRate') +
  theme(legend.position = 'right',
        legend.key.height = unit(0.7, 'cm'),
        text = element_text(size = 24, family = font_family),
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank())

ggsave(filename = str_glue('{BASE_DIRECTORY}/results/figures/pdf/hallucination_by_usdc_geography.pdf'), plot = p2, device = 'pdf', width = 9, height = 7)
