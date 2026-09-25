// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Preston Logan
void exp_dmb(void) { __asm__ volatile("dmb sy" ::: "memory"); }
