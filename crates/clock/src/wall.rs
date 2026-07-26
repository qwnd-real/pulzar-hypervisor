//! Wall-clock time, and the calendar arithmetic that connects it to what
//! firmware reports.
//!
//! Firmware answers `GetTime` with a civil date — year, month, day, hour,
//! minute, second, nanosecond — and how far that reading is from UTC. It is the
//! only absolute time a hypervisor is ever handed: once firmware is gone there
//! is no real-time clock driver, only counters that count. So the reading is
//! turned into one number here, nanoseconds since the Unix epoch in UTC, and
//! from then on wall-clock time is that number plus however much the monotonic
//! clock says has passed.
//!
//! Both directions are needed and both are here: the loader turns firmware's
//! civil date into the number, and every log line turns the number back into a
//! date. The conversions are the standard proleptic-Gregorian ones, which are
//! exact — the calendar repeats every 400 years and 146 097 days, so the whole
//! thing is integer arithmetic with no tables and no loops.
//!
//! # What is representable
//!
//! Nanoseconds in a `u64` reach from 1970 to the year 2554, which is why
//! [`Wall::from_civil`] refuses a reading outside that span rather than
//! wrapping into it. A machine whose real-time clock is unset usually reports
//! something in the 2000s, and one whose battery has died reports 1970 or a
//! year before it; the second of those is refused and reported as having no
//! wall clock at all, which is more useful than a confident wrong answer.

use core::fmt::{self, Display, Formatter};

use crate::units::NANOS_PER_SECOND;

/// Seconds in a minute.
const SECONDS_PER_MINUTE: u64 = 60;

/// Seconds in an hour.
const SECONDS_PER_HOUR: u64 = 60 * SECONDS_PER_MINUTE;

/// Seconds in a day. Leap seconds are not counted: Unix time does not have
/// them, and no clock pulzar reads reports one.
const SECONDS_PER_DAY: u64 = 24 * SECONDS_PER_HOUR;

/// Days in the calendar's 400-year cycle, after which every date, weekday and
/// leap day repeats exactly.
const DAYS_PER_ERA: u64 = 146_097;

/// Days from 0000-03-01 — where the March-based year the algorithms use begins
/// — to the Unix epoch.
const DAYS_BEFORE_EPOCH: u64 = 719_468;

/// The largest offset from UTC ACPI and UEFI admit: a whole day either way.
const MAX_OFFSET_MINUTES: i16 = 1440;

/// The first year a `u64` of nanoseconds since the epoch can express, which is
/// the epoch's own.
const MIN_YEAR: u64 = 1970;

/// The last year one can express. The range ends part-way through it, and the
/// arithmetic below is what refuses the days past the end; this bound is here
/// so that a reading centuries out — which is what an unset real-time clock
/// tends to produce — is refused as the nonsense it is rather than as an
/// overflow.
const MAX_YEAR: u64 = 2554;

/// A point in wall-clock time, as nanoseconds since 1970-01-01T00:00:00Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wall(u64);

impl Wall {
    /// The time `nanos` nanoseconds after the Unix epoch.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Nanoseconds since the Unix epoch, which is how this travels between the
    /// loader and the hypervisor.
    #[must_use]
    pub const fn nanos(self) -> u64 {
        self.0
    }

    /// The time a firmware reading names, or `None` if it names no time at all.
    ///
    /// `None` covers a reading the calendar does not admit — a thirty-first of
    /// February, a twenty-fifth hour — and one outside the representable span,
    /// which is a real-time clock that has lost its battery or one set
    /// centuries ahead. Both are firmware saying something that cannot be
    /// believed, and the answer to that is to have no wall clock rather than a
    /// wrong one.
    #[must_use]
    pub fn from_civil(civil: Civil) -> Option<Self> {
        if !civil.is_valid() {
            return None;
        }
        let days = days_from_civil(
            u64::from(civil.year),
            u64::from(civil.month),
            u64::from(civil.day),
        )?;
        let reading = days * SECONDS_PER_DAY
            + u64::from(civil.hour) * SECONDS_PER_HOUR
            + u64::from(civil.minute) * SECONDS_PER_MINUTE
            + u64::from(civil.second);
        // UEFI defines the field as the minutes that have to be *added* to the
        // reading to reach UTC, so a reading east of Greenwich carries a
        // negative offset. Firmware that would not say is taken at its word as
        // UTC: it is a guess, but the only one available, and the alternative is
        // no wall clock at all.
        let offset = i64::from(civil.utc_offset_minutes.unwrap_or_default())
            * i64::try_from(SECONDS_PER_MINUTE).ok()?;
        reading
            .checked_add_signed(offset)?
            .checked_mul(NANOS_PER_SECOND)?
            .checked_add(u64::from(civil.nanosecond))
            .map(Self)
    }

    /// This time `nanos` nanoseconds later.
    ///
    /// Saturating at the end of the representable span rather than wrapping
    /// back to 1970, for the same reason [`Wall::from_civil`] refuses a date
    /// beyond it.
    #[must_use]
    pub const fn after(self, nanos: u64) -> Self {
        Self(self.0.saturating_add(nanos))
    }
}

impl Display for Wall {
    /// ISO 8601 in UTC, to the nanosecond: `2026-07-25T14:03:21.123456789Z`.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let seconds = self.0 / NANOS_PER_SECOND;
        let (year, month, day) = civil_from_days(seconds / SECONDS_PER_DAY);
        let time = seconds % SECONDS_PER_DAY;
        write!(
            formatter,
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
            time / SECONDS_PER_HOUR,
            time % SECONDS_PER_HOUR / SECONDS_PER_MINUTE,
            time % SECONDS_PER_MINUTE,
            self.0 % NANOS_PER_SECOND,
        )
    }
}

/// A civil date and time as firmware reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    /// The year in full, not since 1900.
    pub year: u16,
    /// Month of the year, 1 through 12.
    pub month: u8,
    /// Day of the month, 1 through the length of that month.
    pub day: u8,
    /// Hour of the day, 0 through 23.
    pub hour: u8,
    /// Minute of the hour, 0 through 59.
    pub minute: u8,
    /// Second of the minute, 0 through 59.
    pub second: u8,
    /// Nanoseconds into the second.
    pub nanosecond: u32,
    /// Minutes that have to be added to this reading to reach UTC, or `None`
    /// where firmware declined to say — which UEFI allows, and means the
    /// reading is local time of an unstated zone.
    pub utc_offset_minutes: Option<i16>,
}

impl Civil {
    /// Whether the calendar admits this reading, and whether it lands in the
    /// span a [`Wall`] can hold.
    fn is_valid(&self) -> bool {
        (MIN_YEAR..=MAX_YEAR).contains(&u64::from(self.year))
            && days_in_month(self.year, self.month)
                .is_some_and(|last| (1..=last).contains(&self.day))
            && self.hour < 24
            && self.minute < 60
            && self.second < 60
            && u64::from(self.nanosecond) < NANOS_PER_SECOND
            && self
                .utc_offset_minutes
                .is_none_or(|offset| (-MAX_OFFSET_MINUTES..=MAX_OFFSET_MINUTES).contains(&offset))
    }
}

/// Days from the Unix epoch to a civil date, or `None` for a date before it.
///
/// The standard algorithm, minus its branches for negative years: the only date
/// this crate converts in this direction is one that has to land at or after
/// the epoch to be representable, so a date before it drops out of the one
/// subtraction that can underflow.
const fn days_from_civil(year: u64, month: u64, day: u64) -> Option<u64> {
    // A March-based year puts the leap day at the end, so February's length
    // never shifts the days before it and no month needs a special case.
    let year = if month <= 2 {
        match year.checked_sub(1) {
            Some(year) => year,
            None => return None,
        }
    } else {
        year
    };
    let era = year / 400;
    let year_of_era = year - era * 400;
    let month_of_year = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_of_year + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    (era * DAYS_PER_ERA + day_of_era).checked_sub(DAYS_BEFORE_EPOCH)
}

/// The civil date `days` days after the Unix epoch, as year, month and day.
///
/// The inverse of [`days_from_civil`], and negative dates are absent for the
/// same reason: a [`Wall`] counts from the epoch, so the day number never goes
/// below it.
const fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let days = days + DAYS_BEFORE_EPOCH;
    let era = days / DAYS_PER_ERA;
    let day_of_era = days - era * DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_of_year = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_of_year + 2) / 5 + 1;
    let month = if month_of_year < 10 {
        month_of_year + 3
    } else {
        month_of_year - 9
    };
    let next_year = if month <= 2 { 1 } else { 0 };
    (year_of_era + era * 400 + next_year, month, day)
}

/// Days in `month` of `year`, or `None` if that is not a month.
const fn days_in_month(year: u16, month: u8) -> Option<u8> {
    Some(match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return None,
    })
}

/// Whether `year` has a twenty-ninth of February.
const fn is_leap_year(year: u16) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

// The calendar, checked where the check costs nothing and cannot be skipped.
// This is the one part of the crate no machine can disagree with — the
// arithmetic is the same everywhere — and the one a wrong answer from would be
// hardest to notice, because an off-by-one here produces a date that looks
// entirely ordinary.
const _: () = {
    // The epoch, a leap day and the day after it, the century that is a leap
    // year and the century that is not, an ordinary date, and the last day the
    // range holds.
    const DATES: [(u64, u64, u64); 7] = [
        (1970, 1, 1),
        (1972, 2, 29),
        (1972, 3, 1),
        (2000, 2, 29),
        (2100, 3, 1),
        (2026, 7, 26),
        (2554, 7, 21),
    ];

    assert!(
        days_from_civil(MIN_YEAR, 1, 1).unwrap() == 0,
        "the epoch has to be day zero"
    );
    assert!(
        days_from_civil(1969, 12, 31).is_none(),
        "the day before the epoch has to be out of reach, not wrapped into range"
    );

    let mut index = 0;
    while index < DATES.len() {
        let (year, month, day) = DATES[index];
        let (again, month_again, day_again) =
            civil_from_days(days_from_civil(year, month, day).unwrap());
        assert!(
            again == year && month_again == month && day_again == day,
            "a date has to survive the trip to a day number and back unchanged"
        );
        index += 1;
    }

    assert!(
        is_leap_year(2024) && is_leap_year(2000) && !is_leap_year(2100) && !is_leap_year(1970),
        "a leap year is every fourth, less every hundredth, plus every four hundredth"
    );
    assert!(
        days_in_month(2000, 2).unwrap() == 29 && days_in_month(2100, 2).unwrap() == 28,
        "february is the month the leap rule reaches"
    );
    assert!(
        days_in_month(2026, 0).is_none() && days_in_month(2026, 13).is_none(),
        "a month outside the year is not a month"
    );
};

// Where [`MAX_YEAR`] comes from. It is not a policy about how far ahead a clock
// may be set: it is the year a `u64` of nanoseconds runs out in, and these two
// assertions are what keep the constant pinned to that fact rather than to
// whoever last edited it.
const _: () = {
    let last = days_from_civil(MAX_YEAR, 1, 1).unwrap() * SECONDS_PER_DAY;
    assert!(
        last.checked_mul(NANOS_PER_SECOND).is_some(),
        "the whole of the last year has to be representable"
    );
    let past = days_from_civil(MAX_YEAR + 1, 1, 1).unwrap() * SECONDS_PER_DAY;
    assert!(
        past.checked_mul(NANOS_PER_SECOND).is_none(),
        "the year after it has to be out of reach"
    );
};
