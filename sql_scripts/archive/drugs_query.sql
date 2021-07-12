select
	ofr.offender_id as offender_id,
	ofr.incident_id as incident_id,
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
	i.cleared_except_id,
	a.arrest_type_id,
	ag.state_abbr,
	concat(ag.pub_agency_name, '-', ag.state_name) as agency,
	inc.drug_offense,
	inc.drug_equipment_offense,
	sus_drug.unique_drug_type_count,
	sus_drug.drug_type_count,
	sus_drug.unique_drug_measure_per_drug,
	sus_drug.crack_qty,
	sus_drug.cocaine_qty,
	sus_drug.heroin_qty,
	sus_drug.cannabis_qty,
	sus_drug.meth_amphetamines_qty,
	sus_drug.other_drugs,
	ca.criminal_act_count,
	ca.criminal_act,
	inc_counts.offense_count,
	inc_counts.offender_count,
	inc_counts.location_category,
	prop.drug_equipment_value
from nibrs_offender ofr
left join nibrs_incident i on ofr.incident_id = i.incident_id
left join nibrs_arrestee a on ofr.incident_id = a.incident_id and ofr.offender_seq_num = a.arrestee_seq_num
left join (
	select 
		count(prop.property_id) as property_count,
		prop.incident_id,
		sum(case
			when npd.prop_desc_id = 11 then npd.property_value
		end) as drug_equipment_value
	from nibrs_property prop
	left join nibrs_property_desc npd on prop.property_id = npd.property_id
	group by prop.incident_id
) prop on ofr.incident_id = prop.incident_id
INNER JOIN (
	SELECT
		o.incident_id,
		max(case 
			when o.offense_type_id = 16 then 1
			else 0
		end) as drug_offense,
		max(case
			when o.offense_type_id = 35 then 1
			else 0
		end) as drug_equipment_offense,
		max(case
			when o.offense_type_id not in (16, 35) then 1
			else 0
		end) as other_offense
	FROM nibrs_offense o
	group by o.incident_id
	) inc ON inc.incident_id = i.incident_id
left join (
	select
		COUNT(distinct sd.suspected_drug_type_id) as unique_drug_type_count,
		COUNT(distinct case
			when sd.suspected_drug_type_id = 5 then sd.drug_measure_type_id
			end) as unique_drug_measure_per_drug,
		COUNT(sd.suspected_drug_type_id) as drug_type_count,
		-- split quantities of 5 most common drugs into new columns
		-- NEEDS CONVERTING TO STANDARD UNITS
		sum(case
			when sd.suspected_drug_type_id = 1 then sd.est_drug_qty
			else 0
		end) as crack_qty,
		sum(case
			when sd.suspected_drug_type_id = 2 then sd.est_drug_qty
			else 0
		end) as cocaine_qty,
		sum(case
			when sd.suspected_drug_type_id = 4 then sd.est_drug_qty
			else 0
		end) as heroin_qty,
		sum(case
			when sd.suspected_drug_type_id = 5 then sd.est_drug_qty
			else 0
		end) as cannabis_qty,
		sum(case
			when sd.suspected_drug_type_id = 12 then sd.est_drug_qty
			else 0
		end) as meth_amphetamines_qty,
		-- bool for other drugs being present too
		count(distinct case
			when sd.suspected_drug_type_id not in (1,2,4,5,12) then sd.suspected_drug_type_id
		end) as other_drugs,
		--
		np.incident_id as incident_id
	FROM nibrs_property np
	    INNER JOIN nibrs_suspected_drug sd
	          ON sd.property_id = np.property_id
	group by np.incident_id 
) sus_drug on sus_drug.incident_id = ofr.incident_id
left join (
	select
		COUNT(distinct ca1.criminal_act_id) as criminal_act_count,
		substring(max(distinct case
			when ca1.criminal_act_id = 8 then '0:consuming'
			when ca1.criminal_act_id = 6 then '1:possessing'
			when ca1.criminal_act_id = 1 then '2:buying'
			when ca1.criminal_act_id = 7 then '3:transporting'
			when ca1.criminal_act_id = 2 then '4:producing'
			when ca1.criminal_act_id = 3 then '5:distributing'
		end), 3) as criminal_act,
		ofr.incident_id
	FROM nibrs_offender ofr
		join nibrs_incident ni on ni.incident_id = ofr.incident_id
		join nibrs_offense no2 on no2.incident_id = ni.incident_id
		join nibrs_criminal_act ca1 on ca1.offense_id = no2.offense_id 
	group by ofr.incident_id 
) ca on ca.incident_id = i.incident_id
left join (
	select
		count(distinct no4.offender_id) as offender_count,
		count(distinct no3.offense_id) as offense_count,
		string_agg(distinct case
			when no3.location_id in (13, 18) then 'street'
			when no3.location_id in (8, 7, 23, 12) then 'store'
			when no3.location_id = 20 then 'home'
			when no3.location_id = 14 then 'hotel/motel'
			when no3.location_id = 41 then 'elementary school'
			else 'other'
		end,
		';') as location_category,
		count(distinct case
			when no3.location_id in (13, 18) then 'street'
			when no3.location_id in (8, 7, 23, 12) then 'store'
			when no3.location_id = 20 then 'home'
			when no3.location_id = 14 then 'hotel/motel'
			when no3.location_id = 41 then 'elementary school'
			else 'other'
		end) as location_count,
		ni2.incident_id
	from nibrs_incident ni2
		join nibrs_offense no3 on no3.incident_id = ni2.incident_id
		join nibrs_offender no4 on no4.incident_id = ni2.incident_id 
	group by ni2.incident_id
) inc_counts on ofr.incident_id = inc_counts.incident_id
left join agencies ag on i.agency_id = ag.agency_id
where other_offense = 0 and inc_counts.location_count = 1

