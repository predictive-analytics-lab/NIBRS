select
--	ofr.offender_id as offender_id,
--	ofr.incident_id as incident_id,
	case
		when ofr.ethnicity_id = 1 then 'hispanic/latino'
		when ofr.race_id = 1 then 'white'
		when ofr.race_id = 2 then 'black'
		when ofr.ethnicity_id is null or ofr.race_id = 0 or ofr.ethnicity_id = 3 then 'unknown'
		else 'other/mixed'
	end as dm_offender_race_ethnicity,
	ofr.sex_code dm_offender_sex,
	ofr.age_num dm_offender_age,
	ofr.offender_seq_num,
--	o.offense_id,
	o.offense_type_id,
	o.location_id,
	i.cleared_except_id,
	a.arrest_type_id,
	ca.criminal_act_id,
--	v.victim_id,
	vor.relationship_id,
--	prop.property_id,
--	prop.prop_loss_id,
--	pd.prop_desc_id,
--	pd.property_value,
--	sd.suspected_drug_type_id,
--	sd.est_drug_qty,
--	sd.drug_measure_type_id,
	property_count,
	drug_type_count,
	ag.state_abbr,
	concat(ag.pub_agency_name, '-', ag.state_name) as agency
from nibrs_offender ofr
join nibrs_offense o on o.incident_id = ofr.incident_id
left join nibrs_incident i on ofr.incident_id = i.incident_id
left join nibrs_arrestee a on ofr.incident_id = a.incident_id and ofr.offender_seq_num = a.arrestee_seq_num
left join nibrs_criminal_act ca on o.offense_id = ca.offense_id
left join nibrs_victim v on ofr.incident_id = v.incident_id
left join nibrs_victim_offender_rel vor on v.victim_id = vor.victim_id and ofr.offender_id = vor.offender_id 
-- left join nibrs_property prop on ofr.incident_id = prop.incident_id -- can have multiple
-- left join nibrs_property_desc pd on prop.property_id = pd.property_id
-- left join nibrs_suspected_drug sd on prop.property_id = sd.property_id -- can have multiple
left join (
	select 
		count(prop.property_id)
		as property_count,
		prop.incident_id
	from nibrs_property prop
	group by prop.incident_id
) prop on ofr.incident_id = prop.incident_id
left join (
	SELECT COUNT(distinct sd.suspected_drug_type_id) as drug_type_count, np.incident_id as incident_id
	FROM nibrs_offender ofr
	    INNER JOIN nibrs_property np on np.incident_id = ofr.incident_id
	    INNER JOIN nibrs_suspected_drug sd
	          ON sd.property_id = np.property_id
	group by np.incident_id 
) sus_drug on sus_drug.incident_id = ofr.incident_id 
left join (
	select 
		count(distinct )
		as property_count,
		prop.incident_id
	from nibrs_property prop
	group by prop.incident_id
) prop on ofr.incident_id = prop.incident_id
left join agencies ag on i.agency_id = ag.agency_id
where o.offense_type_id in (16, 35)


