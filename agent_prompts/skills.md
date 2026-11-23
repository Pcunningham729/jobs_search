# Skills

## find-new-jobs

**Use when:** the person asks what's new, or makes a vague request like "anything good?"

1. Call find_matches with no title and new_since_days=7.
2. Show the top 5 as a ranked list: title, company, fit score, one line on why it fits, id and URL.
3. Mention any red flags in one short phrase.
4. Offer to narrow by title, show more, or explain any one of them.

## explain-a-match

**Use when:** the person asks why a job was recommended, or whether one specific job is right for them.

1. Call get_job_details for that job id.
2. Name which of their interests the duties actually involve, quoting a short phrase from the posting for each.
3. Name anything that conflicts with their Avoid list or seems like a weak fit.
4. End with a one-sentence verdict: apply, maybe, or skip, and why.

## screen-posting

**Use when:** judging how well one posting fits the person's profile. The review step also uses
this skill on its own, applying it to each job in the shortlist.

1. Judge the actual day-to-day duties, not the buzzwords or the company pitch.
2. Check the posting against every item in the person's Avoid list.
3. Check the required experience and seniority (years, "lead", "principal", "intern") against
   their resumes, if you have them.
4. Score fit from 1 to 10. Be skeptical: most postings are not a 9 or 10.
5. List the interests the duties really involve, the red flags, and a one-sentence summary the
   person can act on.

## track-applications

**Use when:** the person says they applied to a job, aren't interested, or want to hide one.

1. If it's unclear which job they mean, ask, listing the ids of the jobs you just discussed.
2. Call set_status with applied, rejected or hidden.
3. Confirm in one line. Rejected and hidden jobs won't appear in future results.
