# `weatherloo` take-home project

Estimated time: 60 minutes

## Instructions

You'll be working with a real-world weather dataset called ERA5 - it's a massive historical record of the Earth's atmosphere. It tracks information like temperature, wind, humidity, (and so much more), measured at thousands of points across the globe, every few hours. ERA5 is our best estimate of what the atmosphere actually looked like at any given moment in time, and it's used as the foundation for many modern AI weather models.

The specific dataset is hosted here:
https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr

Your goal is to pick one of those weather measurements and visualize how it changes over a 24-hour period across the globe.

### Task 1: Choose a weather variable

Pick any weather variable from the dataset that interests you. In a short paragraph (a few sentences is fine!), write an ELI5-style explanation of what it represents physically. Also note whether it's a single-level variable (e.g. surface temperature) or one associated with pressure levels (e.g. wind at different altitudes in the atmosphere).

### Task 2: Plot the variable

Using the dataset above, create an animation of your chosen variable over a 24-hour window. If your variable has multiple pressure levels, just pick one.

How you present the animation is completely up to you - a matplotlib animation, an interactive globe, a small web app, whatever you think is cool. Just include a note on how to run it.

### Submitting

Send us a GitHub repo

## Goals of this project

- Getting comfortable working with real weather data (similar to what you'd work with on the team!)
- Learning something new about the atmosphere
- Showing us how you approach an open-ended problem

## Notes
- We fully support using AI tools — if you do, feel free to attach the prompts or tools you used.
- Don't stress too much about this. There's no single right answer.
- If you have any questions, reach out to @ayaonic or @cindehaa on Discord — we're happy to help!
