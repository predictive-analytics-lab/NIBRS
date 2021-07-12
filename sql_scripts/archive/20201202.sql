select 
	case o.offense_type_id
		when 51 then 'simp_assault'
		when 27 then 'agg_assault'
		when 36 then 'rape'
		when 44 then 'intimidation'
		when 56 then 'fondling'
		else 'other'
	end as offense_type,
	case ni.cleared_except_id
		when 1 then 'other'
		when 3 then 'other'
		when 5 then 'other'
		when 2 then 'declined'
		when 4 then 'uncooperative'
		when 6 then 'not_cleared'
	end as cleared,
	case
		when ofr.age_num < 18 then '<18'
		when ofr.age_num <= 24 then '18-24'
		when ofr.age_num <= 31 then '25-31'
		when ofr.age_num <= 42 then '32-42'
		else '>42'
	end as dm_offender_age,
	ofr.sex_code dm_offender_sex,
	case ofr.race_id 
		when 0 then 'unknown'
		when 1 then 'white'
		when 2 then 'black'
		else 'other'
	end as dm_offender_race,
	case coalesce(ofr.ethnicity_id, 3) 
		when 1 then 'yes'
		when 2 then 'no'
		when 3 then 'unknown'
		when 4 then 'mixed'
	end as dm_offender_hispanic_latino,
	case
		when nv.age_num < 18 then '<18'
		when nv.age_num <= 24 then '18-24'
		when nv.age_num <= 31 then '25-31'
		when nv.age_num <= 42 then '32-42'
		else '>42'
	end as dm_victim_age,
	nv.sex_code dm_victim_sex,
	case nv.race_id
		when 0 then 'unknown'
		when 1 then 'white'
		when 2 then 'black'
		else 'other'
	end as dm_victim_race,
	case coalesce(nv.ethnicity_id, 3)
		when 1 then 'yes'
		when 2 then 'no'
		when 3 then 'unknown'
		when 4 then 'mixed'
	end as dm_victim_hispanic_latino,
	coalesce(nullif(nv.resident_status_code, ''), 'U') dm_victim_residency,
	-- Replace Null or 5 with none; replace 4 with minor; and everything else with major
	-- regexp_replace(replace(replace(coalesce(cast(nvi.injury_id as varchar), '5'), '4', 'minor'), '5', 'none'), '^()$', 'major') injury_severity,
	case coalesce(nvi.injury_severity, '0')
		when '0' then 'none'
		when '1' then 'minor'
		else 'major'
	end
	as injury_severity,
	-- Replace weapons [2,3,4,5,6,21,22,23,24,25]:firearm, [1,26,27,28]:brute_force, [7,8,9,10,12]:object, [19, 20, NULL]:none, everything else:unknown
	--regexp_replace(regexp_replace(regexp_replace(regexp_replace(regexp_replace(coalesce(cast(nw.weapon_id as varchar), '0'), '^(2|3|4|5|6|21|22|23|24|25)$', 'firearm'), '^(1|26|27|28)$', 'brute_force'), '^(7|8|9|10|12)$', 'object'), '^(0|19|20)$', 'none'), '^\d+$', 'other') weapon_type,
	case coalesce(nw.weapon_type, 0) 
		when 0 then 'none'
		when 1 then 'unknown'
		when 2 then 'unarmed'
		when 3 then 'object'
		when 4 then 'firearm'
	end
	as weapon_type,
	-- Replace relationships [1,2,3,4,7,8,9,12,14,16,26,27]:known, [5,6,10,11,13,15,17,19,20,21,22,23]:family, 24:stranger, [NULL, 18, 25]:unknown
	regexp_replace(replace(regexp_replace(regexp_replace(cast(coalesce(nvor.relationship_id, 18) as varchar), '^(1|2|3|4|7|8|9|12|14|16|26|27)$', 'known'), '^(5|6|10|11|13|15|17|19|20|21|22|23)$', 'family'), '24', 'stranger'), '^(18|25)$', 'unknown') relationship_type,
	case when na.arrestee_id is not null
        then 'Yes'
        else 'No'
	end
	as arrest,
	nbm.biased
FROM nibrs_offense o
join nibrs_offense_type ot on ot.offense_type_id = o.offense_type_id
right join (
	select ofr_inner.incident_id
	from nibrs_offender ofr_inner
	group by ofr_inner.incident_id
	having count(ofr_inner.offender_id) = 1
	) ofr_incident_id on ofr_incident_id.incident_id = o.incident_id
left join nibrs_offender ofr on ofr.incident_id = o.incident_id
join nibrs_incident ni on o.incident_id = ni.incident_id
right join (
	select nv_inner.incident_id
	from nibrs_victim nv_inner
	group by nv_inner.incident_id
	having count(nv_inner.victim_id) = 1
	) nv_victim_id on nv_victim_id.incident_id = o.incident_id
join nibrs_victim nv on o.incident_id = nv.incident_id
left join nibrs_arrestee na on o.incident_id = na.incident_id
left join (
	select -- Convert bias_id into a bool 'biased', grouping with BOOL_OR
		bool_or(case when nbm.bias_id = 21 or nbm.bias_id = 22
			then false 
			else true 
		end)
		as biased,
		nbm.offense_id 
	from nibrs_bias_motivation nbm
	group by nbm.offense_id
	)
	nbm on nbm.offense_id = o.offense_id
-- join nibrs_bias_list nbl on nbl.bias_id = nbm.bias_id 
left join nibrs_victim_offender_rel nvor on nvor.offender_id = ofr.offender_id and nvor.victim_id = nv.victim_id 
left join (
	select -- Replace injury IDs with severity, and take maximum where duplicated
		max(translate(cast(nvi.injury_id as varchar), '54123678', '01222222')) injury_severity,
		nvi.victim_id
	from nibrs_victim_injury nvi
	group by nvi.victim_id
) nvi on nvi.victim_id = nv.victim_id
--left join nibrs_weapon nw on nw.offense_id = o.offense_id
left join (
	select 
		max(case -- Replace weapons [2,3,4,5,6,21,22,23,24,25]:4, [1,26,27,28]:2, [7,8,9,10,12]:3, [19, 20, NULL]:0, everything else:1
			when cast(nw.weapon_id as varchar) ~ '^(0|19|20)$' then 0
			when cast(nw.weapon_id as varchar) ~ '^(1|26|27|28)$' then 2
			when cast(nw.weapon_id as varchar) ~ '^(7|8|9|10|12)$' then 3
			when cast(nw.weapon_id as varchar) ~ '^(2|3|4|5|6|21|22|23|24|25)$' then 4
			else 1
		end)
		as weapon_type,
		nw.offense_id
	from nibrs_weapon nw
	group by nw.offense_id 
) nw on nw.offense_id = o.offense_id 
WHERE ot.crime_against = 'Person'
and ofr.age_num is not null
and nv.age_num is not null;
-- Sources of duplicates:
--	- multiple bias (5)
-- 	- multiple injury (~2k)


