# Hack the North 2026 submission pack

Deadline: **July 27, 2026 at 11:59 PM ET**  
Event: **September 18-20, 2026, in person at the University of Waterloo**

## Q1: Something I am genuinely proud of

The thing I am proudest of is EarSight, a machine anomaly detector that learns
what a healthy pump sounds like from normal audio, then flags changes. I did not
want a demo that only looked convincing, so I evaluated it on the full 9,755-clip
MIMII fan/pump dataset. Its cosine-kNN scorer reached fan AUC 0.693 and pump AUC
0.880, above the published DCASE 2020 baselines of 0.658 and 0.729. Reproducibility
is an actual test: the scores must match to three decimals across runs.

The less flattering result mattered too. A separate fan/pump classifier reached
only 0.577 validation accuracy on held-out machine IDs. I documented the
generalization gap instead of hiding it.

I originally built the interface for Even Realities G2 smart glasses and verified
the app in the official simulator. Then I learned that the useful pilot is less
cinematic: a Raspberry Pi box with a microphone and accelerometer that texts a
shop and sends a weekly report. This fall I am aiming to install three boxes on
real Vancouver machines. Building the glasses interface taught me the engineering;
being willing to demote it taught me to follow evidence instead of aesthetics.

## Q1 detail link

Use the public EarSight evidence page once it is deployed. Do not link the private
repository or claim a physical-G2 test until one has actually passed.

## Q2: Goose standoff (49 words)

Same protocol as my pumps: freeze, record thirty seconds of baseline honking, and
wait. The moment the honk deviates, I'll know whether it's bluffing or about to
fail catastrophically. If it charges anyway, I'll log the anomaly, surrender my
sandwich, and accept that the goose passed its stress test.

## Q3: Links

- LinkedIn: add the real profile URL; verify the headline and project dates.
- GitHub: `https://github.com/cwklurks`
- Devpost: leave blank unless it contains a completed project worth opening.
- Other: direct EarSight evidence page first, then `https://connork.com`,
  `https://codesprint.ca`, and `https://huggingface.co/connaaa` if the form allows
  multiple links.

## Honest demo plan

The application does not require a physical-G2 claim. The lowest-risk proof is a
60-90 second controlled playback demo using the official Even Hub simulator and
an abnormal clip from the same MIMII pump ID as the healthy baseline.

1. Open with the live alert state, labeled **controlled playback demo**.
2. Introduce EarSight in one sentence.
3. Show healthy baseline capture, sped up and labeled.
4. Switch to an **abnormal recording from the same pump ID**. Do not call it a
   failing bearing unless that clip carries that label.
5. Keep the transition from LISTENING to SUSPECT/ALERT uncut.
6. Show the exact benchmark comparison and the 0.577 classifier caveat.
7. End on the pivot: glasses are the demo interface; the pilot product is the
   hybrid box, alerts, and weekly report.

If a real G2 hardware run succeeds before the deadline, add a short through-lens
clip and label the companion view **live HUD mirror**. Until then, say only that
the Even Hub app is verified in the official simulator.

## Before submitting

- [ ] Deploy the public evidence page and open it in a logged-out/private browser.
- [ ] Add the deployed evidence URL to Q1 and Other.
- [ ] Record the demo only after one complete rehearsal triggers reliably.
- [ ] Keep playback volume and distance unchanged between healthy and abnormal clips.
- [ ] Add captions and verify the video works without sound.
- [ ] Confirm every link opens without authentication.
- [ ] Remove any claim that physical G2, Pi, accelerometer fusion, pilots, alerts,
      or reports are already verified.
- [ ] Submit before July 27 at 11:59 PM ET; do not plan around the final hour.
