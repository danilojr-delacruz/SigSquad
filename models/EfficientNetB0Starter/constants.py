TARGETS = [
    "seizure_vote",	"lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote"
    ]

# This gives the mapping between labels and their class id
LABEL_TO_ID = {
    "Seizure":0, "LPD" :1, "GPD"  :2,
    "LRDA"   :3, "GRDA":4, "Other":5
}
ID_TO_LABEL = {x:y for y,x in LABEL_TO_ID.items()}