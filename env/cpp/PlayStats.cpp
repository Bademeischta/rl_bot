#include "PlayStats.h"

#include <RLGymSim_CPP/Math.h>

#include <algorithm>
#include <cmath>

namespace RLbot {

int PlayTracker::AttackingThird(float y) {
	if (y > OFF_THIRD_Y)
		return 0;
	if (y < -OFF_THIRD_Y)
		return 1;
	return -1;
}

void PlayTracker::BeginEpisode(const GameState& state) {
	step = 0;
	// Gym::Step schiebt die Arena 1 Tick (tickSkip - actionDelay) vor dem Snapshot weiter
	resetTick = state.lastTickCount > 0 ? state.lastTickCount - 1 : 0;
	kickoffOpen = false;
	kickoffTouchStep = -1;
	spellOpen = false;
	lastShots.clear();
	lastBoost.clear();
	lastSpeed.clear();
}

void PlayTracker::CloseKickoff(PlayEvents& ev) {
	ev.kickoffs.push_back(kickoff);
	kickoffOpen = false;
}

void PlayTracker::CloseSpell(PlayEvents& ev, bool goal) {
	spell.seconds = Seconds(spellLastIn - spellStart + 1);
	spell.goal = goal;
	ev.spells.push_back(spell);
	spellOpen = false;
}

PlayEvents PlayTracker::Step(const GameState& state, bool done) {
	PlayEvents ev;
	if (newEpisode) {
		BeginEpisode(state);
		newEpisode = false;
	}
	step++;

	const Vec& ball = state.ball.pos;
	bool scored = done && RLGSC::Math::IsBallScored(ball);
	if (scored)
		ev.goalTeam = ball.y > 0 ? 0 : 1;   // Ball im orangen Tor (+y) = Tor für Blau

	ev.players = (int)state.players.size();
	for (auto& p : state.players)
		ev.playersPerTeam[TeamOf(p)]++;

	// Ballkontakte und Schüsse. ballTouchedStep zählt jeden Kontakt genau einmal: RocketSim schreibt
	// einen Kontakt mit dem Tick vor dem Weiterzählen, die Fenster zweier Snapshots überlappen nicht.
	for (auto& p : state.players) {
		int team = TeamOf(p);
		const auto& hit = p.carState.ballHitInfo;
		if (p.ballTouchedStep && hit.isValid && hit.tickCountWhenHit >= resetTick)
			ev.touches.push_back({ team, p.carId, !p.carState.isOnGround, hit.ballPos.z, hit.tickCountWhenHit });
		int& prevShots = lastShots[p.carId];
		if (p.matchShots > prevShots)
			ev.shots[team] += p.matchShots - prevShots;
		prevShots = p.matchShots;
	}

	// Anstoß: erste Episode-Schritt mit ruhendem Ball in der Mitte
	if (step == 1) {
		kickoffOpen = std::abs(ball.x) < 1 && std::abs(ball.y) < 1 && ball.z < 100 && state.ball.vel.Length() < 1;
		kickoff = KickoffResult();
		kickoffTouchStep = -1;
	}
	if (kickoffOpen) {
		if (kickoff.firstTeam < 0) {
			for (auto& p : state.players) {
				auto it = lastBoost.find(p.carId);
				float boost = p.boostFraction * 100.f;
				int team = TeamOf(p);
				if (it != lastBoost.end() && it->second > boost && ev.playersPerTeam[team] > 0)
					kickoff.boostUsed[team] += (it->second - boost) / ev.playersPerTeam[team];
			}
		}
		if (kickoff.firstTeam < 0 && !ev.touches.empty()) {
			const TouchEvent* first = nullptr;
			uint64_t firstTick = ~0ULL;
			for (auto& t : ev.touches) {
				if (t.tick < firstTick) {
					firstTick = t.tick;
					first = &t;
				}
			}
			kickoff.firstTeam = first->team;
			kickoff.timeToTouch = (float)(firstTick - resetTick) / 120.f;
			for (auto& p : state.players) {
				auto it = lastSpeed.find(p.carId);
				float speed = it != lastSpeed.end() ? it->second : p.phys.vel.Length();
				if (p.carId == first->carId)
					kickoff.touchSpeed = speed;
				else if (TeamOf(p) != first->team)
					kickoff.loserSpeed = std::max(kickoff.loserSpeed, speed);
			}
			kickoffTouchStep = step;
		}
		if (kickoffTouchStep > 0 && kickoff.ballHalfTeam < 0 && !scored
		    && Seconds(step - kickoffTouchStep) >= KICKOFF_POSSESSION_SECS) {
			kickoff.ballHalfTeam = ball.y > 0 ? 0 : 1;
			float best = 1e30f;
			for (auto& p : state.players) {
				float d = (p.phys.pos - ball).Length();
				if (d < best) {
					best = d;
					kickoff.closerTeam = TeamOf(p);
				}
			}
		}
		if (scored && Seconds(step) <= KICKOFF_GOAL_SECS) {
			kickoff.goalTeam = ev.goalTeam;
			// Tor vor der Besitzmessung: der Schütze hatte den Ball
			if (kickoff.firstTeam >= 0 && kickoff.ballHalfTeam < 0)
				kickoff.ballHalfTeam = kickoff.closerTeam = ev.goalTeam;
		}
		if (done || Seconds(step) >= KICKOFF_GOAL_SECS)
			CloseKickoff(ev);
	}

	// Aufenthalte im Angriffsdrittel
	int third = AttackingThird(ball.y);
	ev.ballThirdTeam = third;
	if (spellOpen) {
		if (third == spell.team)
			spellLastIn = step;
		else if (third >= 0 || Seconds(step - spellLastIn) >= OFF_THIRD_EXIT_SECS)
			CloseSpell(ev, false);
	}
	if (!spellOpen && third >= 0) {
		spellOpen = true;
		spell = OffenseSpell();
		spell.team = third;
		spellStart = spellLastIn = step;
	}
	if (done && spellOpen)
		CloseSpell(ev, scored && ev.goalTeam == spell.team);

	// Team (nur ab zwei Spielern je Team)
	for (int team = 0; team < 2; team++) {
		if (ev.playersPerTeam[team] < 2)
			continue;
		std::vector<const PlayerData*> mates;
		for (auto& p : state.players)
			if (TeamOf(p) == team)
				mates.push_back(&p);
		TeamSample s;
		s.team = team;
		double distSum = 0;
		int pairs = 0, commits = 0;
		for (size_t i = 0; i < mates.size(); i++) {
			for (size_t j = i + 1; j < mates.size(); j++, pairs++)
				distSum += (mates[i]->phys.pos - mates[j]->phys.pos).Length();
			Vec toBall = ball - mates[i]->phys.pos;
			float dist = toBall.Length();
			if (dist < COMMIT_DIST && dist > 1e-3f && mates[i]->phys.vel.Dot(toBall / dist) > COMMIT_SPEED)
				commits++;
			float y = mates[i]->phys.pos.y;
			if (team == 0 ? y < ball.y - BACK_MARGIN : y > ball.y + BACK_MARGIN)
				s.lastBack = true;
		}
		s.mateDist = pairs ? (float)(distSum / pairs) : 0.f;
		s.doubleCommit = commits >= 2;
		ev.teams.push_back(s);
	}

	for (auto& p : state.players) {
		lastBoost[p.carId] = p.boostFraction * 100.f;
		lastSpeed[p.carId] = p.phys.vel.Length();
	}
	if (done)
		newEpisode = true;
	return ev;
}

PlayEvents PlayTracker::Flush() {
	PlayEvents ev;
	if (kickoffOpen)
		CloseKickoff(ev);
	if (spellOpen)
		CloseSpell(ev, false);
	newEpisode = true;
	return ev;
}

} // namespace RLbot
