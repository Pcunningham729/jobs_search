"""One module per job board. Each turns that board's API response into `list[Job]`.

This file makes `fetchers` a package (a folder Python can import from), and
lists every fetcher in one place.
"""

from jobs_search.fetchers.ashby import fetch_ashby
from jobs_search.fetchers.greenhouse import fetch_greenhouse
from jobs_search.fetchers.lever import fetch_lever

# Maps each board's name to its fetch function, so other code can look one up
# by name. For example, FETCHERS["greenhouse"]("gitlab") fetches GitLab's jobs.
# When you write a new fetcher, add it here (and its name to `Source` in models.py).
FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
}
