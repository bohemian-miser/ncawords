# NCA Research Results Ledger

### nca.train_ladder_seed
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| lseed-16k | 0 | 0.0029 | 0.0010 | normal_p=0.25 |
| ladder-seed | 0 | 0.0045 | 0.0024 | normal_p=0.25 |
| base-noise-np025-8k | 0 | 0.0050 | 0.0015 | normal_p=0.25 |
| lseed-np10 | 0 | 0.0060 | 0.0048 | normal_p=0.1 |
| base-noise-np025-16k | 0 | 0.0070 | 0.0037 | normal_p=0.25 |
| bench2-t4 | 0 | 0.0079 | 0.0074 | normal_p=0.25 |
| base-noise-np00-8k | 0 | 0.0098 | 0.0076 | normal_p=0.0 |
| base-noise-np00-16k | 0 | 0.0116 | 0.0046 | normal_p=0.0 |
| lseed-np50 | 0 | 0.0134 | 0.0114 | normal_p=0.5 |
| rho-090 | 0 | 0.0137 | 0.0137 | damage_occasional, normal_p=1.0, rho_target=0.9 |
| rho-100 | 0 | 0.0137 | 0.0137 | damage_occasional, normal_p=1.0, rho_target=1.0 |
| lseed-np10-dmg | 0 | 0.0144 | 0.0069 | damage_occasional, normal_p=0.1 |
| lseed-np50-dmg | 0 | 0.0153 | 0.0082 | damage_occasional, normal_p=0.5 |
| ladder-seed-damage | 0 | 0.0169 | 0.0078 | damage_occasional, normal_p=0.25 |
| adap-plain-16k | 0 | 0.0303 | 0.0268 | adaptive, damage_occasional, normal_p=1.0, rho_target=0.0 |
| base-noise-np05-16k | 0 | nan | 0.0174 | normal_p=0.5 |
| bench2-l4 | 0 | 0.0054 | 0.0054 | normal_p=0.25 |
| base-noise-np05-8k | 0 | 0.0136 | 0.0119 | normal_p=0.5 |
| base-noise-np075-8k | 0 | 0.0143 | 0.0138 | normal_p=0.75 |
| base-plain-comp-16k-dmg | 0 | 0.0198 | 0.0190 | damage_occasional, normal_p=1.0 |
| base-noise-np075-16k | 0 | 0.0211 | 0.0180 | normal_p=0.75 |
| base-plain-comp-16k | 0 | 0.0214 | 0.0201 | normal_p=1.0 |
| base-plain-comp-32k | 0 | 0.0218 | 0.0174 | normal_p=1.0 |
| base-plain-comp-8k | 0 | 0.0232 | 0.0222 | normal_p=1.0 |
| base-plain-comp-8k-dmg | 0 | 0.0257 | 0.0251 | damage_occasional, normal_p=1.0 |
| base-plain-6841-16k | 0 | 0.0273 | 0.0273 | normal_p=1.0 |
| base-plain-6841-16k-dmg | 0 | 0.0273 | 0.0273 | damage_occasional, normal_p=1.0 |
| base-plain-6841-8k | 0 | 0.0273 | 0.0273 | normal_p=1.0 |
| base-plain-6841-8k-dmg | 0 | 0.0273 | 0.0272 | damage_occasional, normal_p=1.0 |
| base-plain-nca-16k | 0 | 0.0420 | 0.0342 | normal_p=1.0 |
| base-plain-nca-16k-dmg | 0 | 0.0420 | 0.0395 | damage_occasional, normal_p=1.0 |
| base-plain-nca-8k | 0 | 0.0420 | 0.0325 | normal_p=1.0 |
| base-plain-nca-8k-dmg | 0 | 0.0420 | 0.0374 | damage_occasional, normal_p=1.0 |

### nca.train_negotiate
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| neg-co-d15 | 0 | 0.0005 | 0.0002 | delta=0.15 |
| neg-co-16k | 0 | 0.0006 | 0.0003 | delta=0.15 |
| neg-co-d25-c5 | 0 | 0.0008 | 0.0006 | delta=0.25 |
| neg3-self | 0 | 0.0014 | 0.0008 | delta=0.2, nucleate_p=0.1, self_p=0.25 |
| neg2-ruthless | 0 | 0.0015 | 0.0008 | delta=0.3, nucleate_p=0.15 |
| neg2-nucleate | 0 | 0.0017 | 0.0006 | delta=0.2, nucleate_p=0.15 |

### nca.train_noise_ladder
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| noise-ladder-fixed | 0 | 0.0001 | 0.0001 | schedule=ladder |
| base-nlad-l-rp30-x2 | 0 | 0.0001 | 0.0001 | replay_p=0.3, schedule=ladder |
| base-nlad-l-rp02 | 0 | 0.0001 | 0.0001 | replay_p=0.2, schedule=ladder |
| base-nlad-l-rp04 | 0 | 0.0001 | 0.0001 | replay_p=0.4, schedule=ladder |
| noise-ladder-adaptive | 0 | 0.0002 | 0.0002 | schedule=ladder |
| nladder-replay | 0 | 0.0002 | 0.0001 | replay_p=0.3, schedule=ladder |
| noise-ladder-jumps-fixed | 0 | 0.0003 | 0.0001 | schedule=ladder+jumps |
| noise-ladder-jumps-adaptive | 0 | 0.0004 | 0.0002 | schedule=ladder+jumps |
| base-nlad-lj-rp02 | 0 | 0.0005 | 0.0001 | replay_p=0.2, schedule=ladder+jumps |
| base-nlad-lj-rp04 | 0 | 0.0005 | 0.0001 | replay_p=0.4, schedule=ladder+jumps |
| nladder-jumps-replay50 | 0 | 0.0005 | 0.0003 | replay_p=0.5, schedule=ladder+jumps |
| nladder-jumps-replay | 0 | 0.0007 | 0.0001 | replay_p=0.3, schedule=ladder+jumps |
| base-nlad-lj-rp30-x2 | 0 | 0.0017 | 0.0001 | replay_p=0.3, schedule=ladder+jumps |

### nca.train_organic_reveal
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| org3-frames20 | 0 | 0.0202 | 0.0017 | letter_w=8.0, rot_mode=none |
| org2-frames40 | 0 | 0.0368 | 0.0018 | letter_w=8.0, rot_mode=none |
| org-late20-v2 | 0 | 0.0389 | 0.0018 | letter_w=8.0, rot_mode=late |
| base-org-late20 | 0 | 0.0403 | 0.0018 | letter_w=8.0, rot_mode=late |
| org2-late20-16k | 0 | 0.0456 | 0.0018 | letter_w=8.0, rot_mode=late |
| org3-frames40-dfs | 0 | 0.0516 | 0.0004 | letter_w=8.0, rot_mode=none |
| org2-late45 | 0 | 0.0523 | 0.0018 | letter_w=8.0, rot_mode=late |
| adap-org-frames40 | 0 | 0.0529 | 0.0018 | adaptive, letter_w=8.0, rot_mode=none |
| organic-reveal | 0 | 0.0603 | 0.0021 |  |
| org3-frames40-16k | 0 | 0.0614 | 0.0018 | letter_w=8.0, rot_mode=none |
| base-org-lw2-bfs | 0 | 0.0619 | 0.0019 | letter_w=2.0, rot_mode=aug90 |
| base-org-norot | 0 | 0.0665 | 0.0018 | letter_w=8.0, rot_mode=none |
| org2-nca | 0 | 0.0713 | 0.0015 | letter_w=8.0, rot_mode=none |
| org2-lw8-16k | 0 | 0.0817 | 0.0018 | letter_w=8.0, rot_mode=none |
| org-lw4-bfs | 0 | 0.0874 | 0.0019 | letter_w=4.0, rot_mode=aug90 |
| organic-reveal-norot | 0 | 0.0889 | 0.0027 | rot_mode=none |
| organic-reveal-late20 | 0 | 0.0936 | 0.0027 | rot_mode=late |
| base-org-frames40 | 0 | 0.1020 | 0.0019 | letter_w=8.0, rot_mode=aug90 |
| organic-reveal-dfs | 0 | 0.1101 | 0.0006 | rot_mode=aug90 |
| base-org-nca | 0 | 0.1244 | 0.0017 | letter_w=8.0, rot_mode=aug90 |
| base-org-16k | 0 | 0.1362 | 0.0019 | letter_w=8.0, rot_mode=aug90 |
| org2-life24 | 0 | 0.1738 | 0.0004 | letter_w=8.0, lifespan=24, rot_mode=none |
| base-org-lw8-bfs | 0 | 0.1838 | 0.0019 | letter_w=8.0, rot_mode=aug90 |
| base-org-frames80 | 0 | 0.2061 | 0.0019 | letter_w=8.0, rot_mode=aug90 |
| base-org-life12 | 0 | 0.2187 | 0.0006 | letter_w=8.0, lifespan=12, rot_mode=aug90 |
| org2-dfs | 0 | 0.2474 | 0.0004 | letter_w=8.0, rot_mode=none |
| base-org-lw8-dfs | 0 | 0.2980 | 0.0006 | letter_w=8.0, rot_mode=aug90 |
| base-org-life24 | 0 | 0.3000 | 0.0006 | letter_w=8.0, lifespan=24, rot_mode=aug90 |
| org2-life12 | 0 | 0.3217 | 0.0004 | letter_w=8.0, lifespan=12, rot_mode=none |
| organic-lifespan-n12 | 0 | 0.3327 | 0.0006 | letter_w=8.0, lifespan=12, rot_mode=aug90 |
| org2-life6 | 0 | 0.3589 | 0.0004 | letter_w=8.0, lifespan=6, rot_mode=none |
| org-frames40 | 0 | 0.3713 | 0.0025 | letter_w=8.0, rot_mode=aug90 |
| org-norot-v2 | 0 | 0.3967 | 0.0180 | letter_w=8.0, rot_mode=none |
| base-org-life6 | 0 | 0.4509 | 0.0006 | letter_w=8.0, lifespan=6, rot_mode=aug90 |
| org-life6-v2 | 0 | 0.5282 | 0.0006 | letter_w=8.0, lifespan=6, rot_mode=aug90 |
| organic-reveal-lw8 | 0 | 0.5719 | 0.0121 | letter_w=8.0, rot_mode=aug90 |
| org-16k | 0 | 0.5873 | 0.0026 | letter_w=8.0, rot_mode=aug90 |
| org-life12-v2 | 0 | 0.6651 | 0.0006 | letter_w=8.0, lifespan=12, rot_mode=aug90 |
| org-lw8-warmup | 0 | 0.7077 | 0.0026 | letter_w=8.0, rot_mode=aug90 |
| org-text-nca | 0 | 0.7100 | 0.0096 | letter_w=8.0, rot_mode=aug90 |
| org-frames120 | 0 | 0.7138 | 0.0006 | letter_w=8.0, rot_mode=aug90 |
| org-lw16-bfs | 0 | 0.7656 | 0.0026 | letter_w=16.0, rot_mode=aug90 |
| org-life12-bfs | 0 | 0.7663 | 0.0121 | letter_w=8.0, lifespan=12, rot_mode=aug90 |
| org-life24-v2 | 0 | 0.8359 | 0.0006 | letter_w=8.0, lifespan=24, rot_mode=aug90 |
| organic-lifespan-n6 | 0 | 0.8436 | 0.0006 | letter_w=8.0, lifespan=6, rot_mode=aug90 |
| organic-reveal-dfs2 | 0 | 0.8591 | 0.0006 | letter_w=8.0, rot_mode=aug90 |
| org-lw8-dfs | 0 | 0.8591 | 0.0006 | letter_w=8.0, rot_mode=aug90 |
| organic-lifespan-n24 | 0 | 0.8638 | 0.0006 | letter_w=8.0, lifespan=24, rot_mode=aug90 |
| org-lw16-dfs | 0 | 1.0908 | 0.0006 | letter_w=16.0, rot_mode=aug90 |

### nca.train_slime
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| base-slime-sd3-evap06 | 0 | 0.0433 | 0.0147 |  |
| food-co-w10 | 0 | 0.0605 | 0.0482 | food_text=CO |
| slime-evap24 | 0 | 0.0649 | 0.0538 |  |
| slime-agents2k | 0 | 0.0722 | 0.0616 |  |
| base-slime-sd3-evap24 | 0 | 0.0792 | 0.0468 |  |
| food-co-w06 | 0 | 0.0806 | 0.0559 | food_text=CO |
| food-co-evap24 | 0 | 0.0848 | 0.0436 | food_text=CO |
| slime-sub6 | 0 | 0.0942 | 0.0115 |  |
| food-nca-w08 | 0 | 0.0982 | 0.0491 | food_text=NCA |
| slime-nca | 0 | 0.1013 | 0.0390 |  |
| base-slime-sd9-evap24 | 0 | 0.1055 | 0.0674 |  |
| base-slime-sd9-rng1 | 0 | 0.1347 | 0.0386 |  |
| slime-evap06 | 0 | 0.1388 | 0.0252 |  |
| base-slime-16k-sd9 | 0 | 0.1424 | 0.0367 |  |
| slime-sd3 | 0 | 0.1446 | 0.0590 |  |
| slime-sd9 | 0 | 0.1570 | 0.0285 |  |
| base-slime-rng1 | 0 | 0.1590 | 0.0549 |  |
| slime-agents8k | 0 | 0.1712 | 0.0045 |  |
| slime-16k | 0 | 0.1713 | 0.0451 |  |
| base-slime-frames240 | 0 | 0.1714 | 0.0107 |  |

### nca.train_staged
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| staged2-co | 0 | 0.0001 | 0.0001 | replay_p=0.3 |
| staged-comp-24k | 0 | 0.0003 | 0.0001 |  |
| staged-co | 0 | 0.0006 | 0.0001 |  |
| staged-comp | 0 | 0.0006 | 0.0003 |  |
| staged2-comp | 0 | 0.0008 | 0.0002 | replay_p=0.3 |

### nca.train_web_9_line
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| base-9line-nca-single-16k | 0 | 0.0000 | 0.0000 |  |
| base-9line-nca-noise-16k | 0 | 0.0001 | 0.0000 |  |
| base-9line-nca-single-8k | 0 | 0.0001 | 0.0001 |  |
| base-9line-nca-noise-8k | 0 | 0.0002 | 0.0002 |  |
| base-9line-6841-single-16k | 0 | 0.0006 | 0.0003 |  |
| base-9line-6841-noise-8k | 0 | 0.0019 | 0.0014 |  |
| base-9line-6841-noise-16k | 0 | 0.0020 | 0.0008 |  |
| base-9line-comp-noise-16k | 0 | 0.0024 | 0.0003 |  |
| base-9line-comp-single-8k | 0 | 0.0026 | 0.0018 |  |
| base-9line-6841-single-8k | 0 | 0.0029 | 0.0029 |  |
| base-9line-comp-noise-8k | 0 | 0.0033 | 0.0007 |  |
| base-9line-comp-single-16k | 0 | 0.0040 | 0.0007 |  |

### nca.train_web_evaporate
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| base-evap-comp-8k | 0 | 0.0044 | 0.0012 |  |
| base-evap-nca-16k | 0 | 0.0045 | 0.0002 |  |
| base-evap-nca-8k | 0 | 0.0053 | 0.0010 |  |
| base-evap-6841-8k | 0 | 0.0136 | 0.0035 |  |
| base-evap-6841-16k | 0 | 0.0200 | 0.0026 |  |

### nca.train_web_hidden
| Run | Steps | Final Loss | Min Loss | Notable Args |
|-----|-------|------------|----------|---------------|
| base-hid-comp-single-32k | 0 | 0.0173 | 0.0054 |  |
| base-hid-6841-single-16k | 0 | 0.0297 | 0.0297 |  |
| base-hid-nca-single-16k | 0 | 0.0298 | 0.0298 |  |
| base-hid-comp-noise-8k | 0 | 0.0298 | 0.0298 |  |
| base-hid-comp-single-16k | 0 | 0.0300 | 0.0298 |  |
| base-hid-comp-noise-16k | 0 | 0.0301 | 0.0298 |  |
| base-hid-6841-noise-16k | 0 | 0.0303 | 0.0299 |  |
| base-hid-6841-single-8k | 0 | 0.0438 | 0.0300 |  |
| base-hid-6841-noise-8k | 0 | 0.0438 | 0.0299 |  |
| base-hid-nca-single-8k | 0 | 0.0444 | 0.0298 |  |
| base-hid-nca-noise-8k | 0 | 0.0444 | 0.0298 |  |
| smoke-full-loop | 0 | 0.0471 | 0.0336 |  |
| base-hid-nca-noise-16k | 0 | N/A | N/A |  |
