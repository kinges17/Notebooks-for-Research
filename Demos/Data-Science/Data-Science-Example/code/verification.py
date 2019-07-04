import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Contains lists that define the schema of the data and the expected format

# Define expected column types: please view the survey data for more detail
date_col = 'DATE.SUBMITTED'

numerical_cols = ['RESPONSE.ID', 'PARTICIPATION.TYPE.FOLLOW', 'PARTICIPATION.TYPE.USE.APPLICATIONS',
                  'PARTICIPATION.TYPE.USE.DEPENDENCIES', 'PARTICIPATION.TYPE.CONTRIBUTE',
                  'PARTICIPATION.TYPE.OTHER', 'PARTICIPATION.TYPE.ANY.REPONSE', 'TRANSLATED']

bool_cols = ['INFO.JOB', 'RECEIVED.HELP', 'PROVIDED.HELP', 'DISCOURAGING.BEHAVIOR.UNWELCOMING.LANGUAGE',
             'DISCOURAGING.BEHAVIOR.REJECTION.WOUT.EXPLANATION', 'DISCOURAGING.BEHAVIOR.DISMISSIVE.RESPONSE',
             'DISCOURAGING.BEHAVIOR.BAD.DOCS', 'DISCOURAGING.BEHAVIOR.CONFLICT',
             'DISCOURAGING.BEHAVIOR.LACK.OF.RESPONSE']

string_cols = ['STATUS', 'CONTRIBUTOR.TYPE.CONTRIBUTE.CODE', 'CONTRIBUTOR.TYPE.CONTRIBUTE.DOCS', 'OSS.AS.JOB',
               'CONTRIBUTOR.TYPE.PROJECT.MAINTENANCE', 'CONTRIBUTOR.TYPE.FILE.BUGS', 'SEEK.OPEN.SOURCE',
               'CONTRIBUTOR.TYPE.FEATURE.REQUESTS', 'CONTRIBUTOR.TYPE.COMMUNITY.ADMIN', 'EMPLOYMENT.STATUS',
               'PROFESSIONAL.SOFTWARE', 'FUTURE.CONTRIBUTION.INTEREST', 'FUTURE.CONTRIBUTION.LIKELIHOOD',
               'OSS.USER.PRIORITIES.LICENSE', 'OSS.USER.PRIORITIES.CODE.OF.CONDUCT', 'EXTERNAL.EFFICACY',
               'OSS.USER.PRIORITIES.CONTRIBUTING.GUIDE', 'OSS.CONTRIBUTOR.PRIORITIES.WELCOMING.COMMUNITY',
               'OSS.USER.PRIORITIES.ACTIVE.DEVELOPMENT', 'OSS.USER.PRIORITIES.RESPONSIVE.MAINTAINERS',
               'OSS.USER.PRIORITIES.WELCOMING.COMMUNITY', 'OSS.USER.PRIORITIES.WIDESPREAD.USE', 'OSS.STABILITY',
               'OSS.CONTRIBUTOR.PRIORITIES.LICENSE', 'OSS.CONTRIBUTOR.PRIORITIES.CODE.OF.CONDUCT', 'OSS.HIRING',
               'OSS.CONTRIBUTOR.PRIORITIES.CONTRIBUTING.GUIDE', 'OSS.CONTRIBUTOR.PRIORITIES.CLA', 'FIND.HELPEE',
               'OSS.CONTRIBUTOR.PRIORITIES.ACTIVE.DEVELOPMENT',  'OSS.UX', 'OSS.SECURITY', 'INTERNAL.EFFICACY',
               'OSS.CONTRIBUTOR.PRIORITIES.RESPONSIVE.MAINTAINERS', 'OSS.USER.PRIORITIES.CLA', 'OSS.AT.WORK',
               'OSS.CONTRIBUTOR.PRIORITIES.WIDESPREAD.USE', 'USER.VALUES.REPLICABILITY', 'OSS.IDENTIFICATION',
               'USER.VALUES.STABILITY', 'USER.VALUES.INNOVATION', 'USER.VALUES.SECURITY', 'USER.VALUES.COST',
               'INFO.AVAILABILITY', 'USER.VALUES.COMPATIBILITY', 'USER.VALUES.TRANSPARENCY', 'OSS.IP.POLICY',
               'USER.VALUES.USER.EXPERIENCE', 'USER.VALUES.CUSTOMIZABILITY', 'TRANSPARENCY.PRIVACY.BELIEFS',
               'USER.VALUES.SUPPORT', 'USER.VALUES.TRUSTED.PRODUCER', 'TRANSPARENCY.PRIVACY.PRACTICES.OSS',
               'TRANSPARENCY.PRIVACY.PRACTICES.GENERAL', 'HELPEE.PRIOR.RELATIONSHIP', 'MINORITY.HOMECOUNTRY',
               'FIND.HELPER', 'HELPER.PRIOR.RELATIONSHIP', 'RECEIVED.HELP.TYPE', 'PROVIDED.HELP.TYPE',
               'EMPLOYER.POLICY.APPLICATIONS', 'EMPLOYER.POLICY.DEPENDENCIES', 'GENDER', 'TRANSGENDER.IDENTITY',
               'MINORITY.CURRENT.COUNTRY', 'IMMIGRATION', 'SEXUAL.ORIENTATION', 'WRITTEN.ENGLISH', 'AGE',
               'FORMAL.EDUCATION', 'PARENTS.FORMAL.EDUCATION', 'AGE.AT.FIRST.COMPUTER.INTERNET', 'POPULATION',
               'LOCATION.OF.FIRST.COMPUTER.INTERNET', 'OFF.SITE.ID']


# Define expected column values: please view questionnaire.txt for more detail
expect_0_1 = [0, 1]
expect_completeness = ["Partial", "Complete"]
expect_oft_w_occasionally = ["Never", "Rarely", "Occasionally", "Frequently"]
expect_of_w_sometimes = ["Never", "Rarely", "Sometimes", "Frequently"]
expect_oft_w_always = ["Always", "Sometimes", "Rarely", "Never"]
expect_employment = ["Employed full time", "Employed part time", "Full time student", "Temporarily not working",
                     "Retired or permanently not working (e.g. due to disability)", "Other - please describe"]
expect_agreement = ["Strongly agree", "Somewhat agree", "Neither agree nor disagree",
                    "Somewhat disagree", "Strongly disagree"]
expect_interest = ["Very interested", "Somewhat interested", "Not too interested", "Not at all interested"]
expect_likelihood = ["Very likely", "Somewhat likely", "Somewhat unlikely", "Very unlikely"]
expect_importance = ["Very important to have", "Somewhat important to have", "Not important either way",
                     "Somewhat important not to have", "Very important not to have", "Don't know what this is"]
expect_usability = ["Generally easier to use", "Generally harder to use", "About the same"]
expect_security = ["Generally more secure", "Generally less secure", "About the same"]
expect_stability = ["Generally more stable", "Generally less stable", "About the same"]
expect_soft_importance = ["Extremely important", "Very important", "Moderately important",
                          "Slightly important", "Not at all important"]
expect_attribution = ["Records of authorship should be required so that end users know who created the source code they are working with.",
                      "People should be able to contribute code without attribution, if they wish to remain anonymous."]
expect_info_avail = ["A lot of information about me", "Some information about me",
                     "A little information about me", "No information at all about me"]
expect_y_n = ["Yes", "No"]
expect_y_n_not_sure = ["Yes", "No", "Not sure", "Prefer not to say"]
expect_contr_transp = ["I include my real name.",
                       "I usually contribute using a consistent pseudonym that is easily linked to my real name online.",
                       "I usually contribute using a consistent pseudonym that is not linked anywhere with my real name online.",
                       "I usually contribute using a consistent pseudonym that is not linked anywhere with my real name online. ",
                       "I take precautions to use different usernames in different projects.",
                       "I take precautions to use different usernames  in different projects."]
expect_gen_transp = ["I include my real name.", "I don't publish this kind of content online.",
                     "I usually use a consistent pseudonym that is easily linked to my real name online.",
                     "I usually use a consistent pseudonym that is not linked anywhere with my real name online.",
                     "I take precautions to use different pseudonymns on different platforms."]
expect_find_help = ["I asked for help in a public forum (e.g. in a GitHub Issue, project mailing list, etc.) and someone responded.",
                    "I asked a specific person for help.", "Someone offered me unsolicited help.",
                    "Other - Please describe"]
expect_relationship = ["We knew each other well.", "We knew each other a little.", "We knew each other  a little.",
                       "I knew of them through their contributions to projects, but didn't know them personally.",
                       "Total strangers, I didn't know of them previously."]
expect_help_with = ["Writing code or otherwise implementing ideas.", "Installing or using an application.",
                    "Understanding community norms (e.g. how to submit a contribution, how to communicate effectively).",
                    "Introductions to other people", "Introductions to other people.", "Other (please describe)"]
expect_offer_help = ["They asked for help in a public forum (e.g. in a GitHub Issue, project mailing list, etc.) and I responded.",
                     "They asked me directly for help.", "I reached out to them to offer unsolicited help.",
                     "Other (please describe)"]
# in secondary csv file
expect_neg_behav = ["Hostility or rudeness", "Name calling", "Threats of violence", "Impersonation",
                    "Harassment over a sustained period", "Unsolicited sexual advances or comments",
                    "Malicious publication of personal information (doxxing)", "Stalking",
                    "Harassment across multiple platforms", "Other (please describe)",
                    "Stereotyping based on perceived demographic characteristics", "None of the above"]
# in secondary csv file
expect_neg_resp = ["Asked the user(s) to stop the harassing behavior", "Blocked the user(s) harassing me",
                   "Solicited support from other community members", "Consulted legal counsel/ an attorney",
                   "Reported the incident to project maintainers", "Contacted law enforcement",
                   "Reported the incident to the hosting service or ISP", "Other (please describe)",
                   "I did not react / ignored the incident"]
# in secondary csv file
expect_effective = ["Not at all effective", "A little effective", "Somewhat effective",
                    "Mostly effective", "Completely effective"]
# in secondary csv file
expect_neg_conseq = ["Stopped contributing to a project", "Started contributing under a pseudonym",
                     "Worked, asked questions, or collaborated in private channels more often",
                     "Changed or deleted a username", "Removed or changed content on my public online presence",
                     "Suggested the creation or modification of a Code of Conduct",
                     "Engaged in private discussion with community members about the issue",
                     "Engaged in public discussion with community members about the issue",
                     "Made changes in my life offline (e.g. stopped attending meetups or conferences, etc.)",
                     "Other (please describe)", "None of the above"]
expect_contrib = ["Yes, directly-  some or all of my work duties include contributing to open source projects.",
                  "Yes, indirectly- I contribute to open source in carrying out my work duties, but I am not required or expected to do so.",
                  "No."]
expect_emp_contrib = ["I am not permitted to contribute to open source at all.",
                      "I am permitted to contribute to open source, but need to ask for permission.",
                      "I am free to contribute without asking for permission.", "I'm not sure.",
                      "My employer doesn't have a clear policy on this.", "Not applicable"]
expect_emp_apps = ["Use of open source applications is encouraged.",
                   "Use of open source applications is acceptable if it is the most appropriate tool.",
                   "Use of open source applications is rarely, if ever, permitted.", "I'm not sure.",
                   "My employer doesn't have a clear policy on this.", "Not applicable"]
expect_emp_depends = ["Use of open source dependencies is encouraged.",
                      "Use of open source dependencies is acceptable if it is the most appropriate tool.",
                      "Use of open source dependencies is rarely, if ever permitted.", "I'm not sure.",
                      "My employer doesn't have a clear policy on this.", "Not applicable"]
expect_hiring_imp = ["Very important", "Somewhat important", "Not too important", "Not at all important",
                     "Not applicable-I hadn't made any contributions when I got this job."]
expect_immig = ["Yes, and I intend to stay permanently.", "Yes, and I intend to stay temporarily.",
                "Yes, and I am not sure about my future plans.", "No, I live in the country where I was born."]
expect_gender = ["Man", "Woman", "Non-binary  or Other", "Prefer not to say"]
expect_english_fl = ["Very well", "Moderately well", "Not very well", "Not at all"]
expect_age_range_l = ["17 or younger", "18 to 24 years", "25 to 34 years", "35 to 44 years", "45 to 54 years",
                      "55 to 64 years", "65 to 74 years", "75 years or older", "Prefer not to say"]
expect_age_range_s = ["Younger than 13 years old", "13 - 17 years old", "18 - 24 years old",
                      "25 - 45 years old", "Older than 45 years old"]
expect_edu_level = ["Less than secondary (high) school", "Secondary (high) school graduate or equivalent",
                    "Some college, no degree", "Vocational/trade program or apprenticeship",
                    "Bachelor's degree", "Master's degree",
                    "Doctorate (Ph.D.) or other advanced degree (e.g. M.D., J.D.)"]
expect_loc_first = ["At home (belonging to me or a family member)", "At an internet cafe or similar space",
                    "In a classroom, computer lab, or library at school", "At work (recoded from open ended)",
                    "At a public library or community center", "Other (please describe)"]
expect_population = ["github", "off site community"]
expect_off_s_id = ["off_site_5", "off_site_4", "off_site_6", "off_site_1", "off_site_8", "off_site_9",
                   "off_site_3", "off_site_7", "off_site_2"]

# Maps names of categorical columns to list of expected values
cols_exp_values = [("STATUS", ["Partial", "Complete"]), ("PARTICIPATION.TYPE.FOLLOW", expect_0_1),
                   ("PARTICIPATION.TYPE.USE.APPLICATIONS", expect_0_1),
                   ("PARTICIPATION.TYPE.USE.DEPENDENCIES", expect_0_1),
                   ("PARTICIPATION.TYPE.CONTRIBUTE", expect_0_1), ("PARTICIPATION.TYPE.OTHER", expect_0_1),
                   ("CONTRIBUTOR.TYPE.CONTRIBUTE.CODE", expect_oft_w_occasionally),
                   ("CONTRIBUTOR.TYPE.CONTRIBUTE.DOCS", expect_oft_w_occasionally),
                   ("CONTRIBUTOR.TYPE.PROJECT.MAINTENANCE", expect_oft_w_occasionally),
                   ("CONTRIBUTOR.TYPE.FILE.BUGS", expect_oft_w_occasionally),
                   ("CONTRIBUTOR.TYPE.FEATURE.REQUESTS", expect_oft_w_occasionally),
                   ("CONTRIBUTOR.TYPE.COMMUNITY.ADMIN", expect_oft_w_occasionally),
                   ("EMPLOYMENT.STATUS", expect_employment), ("PROFESSIONAL.SOFTWARE", expect_oft_w_occasionally),
                   ("FUTURE.CONTRIBUTION.INTEREST", expect_interest),
                   ("FUTURE.CONTRIBUTION.LIKELIHOOD", expect_likelihood),
                   ("OSS.USER.PRIORITIES.LICENSE", expect_importance),
                   ("OSS.USER.PRIORITIES.CODE.OF.CONDUCT", expect_importance),
                   ("OSS.USER.PRIORITIES.CONTRIBUTING.GUIDE", expect_importance),
                   ("OSS.USER.PRIORITIES.CLA", expect_importance),
                   ("OSS.USER.PRIORITIES.ACTIVE.DEVELOPMENT", expect_importance),
                   ("OSS.USER.PRIORITIES.RESPONSIVE.MAINTAINERS", expect_importance),
                   ("OSS.USER.PRIORITIES.WELCOMING.COMMUNITY", expect_importance),
                   ("OSS.USER.PRIORITIES.WIDESPREAD.USE", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.LICENSE", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.CODE.OF.CONDUCT", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.CONTRIBUTING.GUIDE", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.CLA", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.ACTIVE.DEVELOPMENT", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.RESPONSIVE.MAINTAINERS", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.WELCOMING.COMMUNITY", expect_importance),
                   ("OSS.CONTRIBUTOR.PRIORITIES.WIDESPREAD.USE", expect_importance),
                   ("SEEK.OPEN.SOURCE", expect_oft_w_always), ("OSS.UX", expect_usability),
                   ("OSS.SECURITY", expect_security), ("OSS.STABILITY", expect_stability),
                   ("INTERNAL.EFFICACY", expect_agreement), ("EXTERNAL.EFFICACY", expect_agreement),
                   ("OSS.IDENTIFICATION", expect_agreement),
                   ("USER.VALUES.STABILITY", expect_soft_importance),
                   ("USER.VALUES.INNOVATION", expect_soft_importance),
                   ("USER.VALUES.REPLICABILITY", expect_soft_importance),
                   ("USER.VALUES.COMPATIBILITY", expect_soft_importance),
                   ("USER.VALUES.SECURITY", expect_soft_importance),
                   ("USER.VALUES.COST", expect_soft_importance),
                   ("USER.VALUES.TRANSPARENCY", expect_soft_importance),
                   ("USER.VALUES.USER.EXPERIENCE", expect_soft_importance),
                   ("USER.VALUES.CUSTOMIZABILITY", expect_soft_importance),
                   ("USER.VALUES.SUPPORT", expect_soft_importance),
                   ("USER.VALUES.TRUSTED.PRODUCER", expect_soft_importance),
                   ("TRANSPARENCY.PRIVACY.BELIEFS", expect_attribution),
                   ("INFO.AVAILABILITY", expect_info_avail), ("INFO.JOB", expect_y_n),
                   ("TRANSPARENCY.PRIVACY.PRACTICES.GENERAL", expect_gen_transp),
                   ("TRANSPARENCY.PRIVACY.PRACTICES.OSS", expect_contr_transp),
                   ("RECEIVED.HELP", expect_y_n), ("FIND.HELPER", expect_find_help),
                   ("HELPER.PRIOR.RELATIONSHIP", expect_relationship),
                   ("RECEIVED.HELP.TYPE", expect_help_with),
                   ("PROVIDED.HELP", expect_y_n), ("FIND.HELPEE", expect_offer_help),
                   ("HELPEE.PRIOR.RELATIONSHIP", expect_relationship),
                   ("PROVIDED.HELP.TYPE", expect_help_with),
                   ("DISCOURAGING.BEHAVIOR.LACK.OF.RESPONSE", expect_y_n),
                   ("DISCOURAGING.BEHAVIOR.REJECTION.WOUT.EXPLANATION", expect_y_n),
                   ("DISCOURAGING.BEHAVIOR.DISMISSIVE.RESPONSE", expect_y_n),
                   ("DISCOURAGING.BEHAVIOR.BAD.DOCS", expect_y_n),
                   ("DISCOURAGING.BEHAVIOR.CONFLICT", expect_y_n),
                   ("DISCOURAGING.BEHAVIOR.UNWELCOMING.LANGUAGE", expect_y_n),
                   # Further data here is in secondary csv file
                   ("OSS.AS.JOB", expect_contrib), ("OSS.AT.WORK", expect_of_w_sometimes),
                   ("OSS.IP.POLICY", expect_emp_contrib),
                   ("EMPLOYER.POLICY.APPLICATIONS", expect_emp_apps),
                   ("EMPLOYER.POLICY.DEPENDENCIES", expect_emp_depends),
                   ("OSS.HIRING", expect_hiring_imp), ("IMMIGRATION", expect_immig),
                   ("MINORITY.HOMECOUNTRY", expect_y_n_not_sure),
                   ("MINORITY.CURRENT.COUNTRY", expect_y_n_not_sure),
                   ("GENDER", expect_gender), ("TRANSGENDER.IDENTITY", expect_y_n_not_sure),
                   ("SEXUAL.ORIENTATION", expect_y_n_not_sure),
                   ("WRITTEN.ENGLISH", expect_english_fl), ("AGE", expect_age_range_l),
                   ("FORMAL.EDUCATION", expect_edu_level),
                   ("PARENTS.FORMAL.EDUCATION", expect_edu_level),
                   ("AGE.AT.FIRST.COMPUTER.INTERNET", expect_age_range_s),
                   ("LOCATION.OF.FIRST.COMPUTER.INTERNET", expect_loc_first),
                   ("PARTICIPATION.TYPE.ANY.REPONSE", expect_0_1),
                   ("POPULATION", expect_population),
                   ("OFF.SITE.ID", expect_off_s_id), ("TRANSLATED", expect_0_1)]


# List of dependencies for columns
dep_contribute = [("PARTICIPATION.TYPE.CONTRIBUTE", 1)]
dep_employed = [("EMPLOYMENT.STATUS", "Employed full time"),
                ("EMPLOYMENT.STATUS", "Employed part time")]
dep_user_priors = [("PARTICIPATION.TYPE.USE.APPLICATIONS", 1),
                   ("PARTICIPATION.TYPE.USE.DEPENDENCIES", 1),
                   ("PARTICIPATION.TYPE.CONTRIBUTE", 1)]
dep_info_job = [("EMPLOYMENT.STATUS", "Employed full time"),
                ("EMPLOYMENT.STATUS", "Employed part time"),
                ("EMPLOYMENT.STATUS", "Full time student"),
                ("EMPLOYMENT.STATUS", "Temporarily not working"),
                ("EMPLOYMENT.STATUS", "Other - please describe")]
dep_rec_help = [("RECEIVED.HELP", "Yes")]
dep_prov_help = [("PROVIDED.HELP", "No")]
dep_oss_job = [("PARTICIPATION.TYPE.CONTRIBUTE", 1)]
dep_emp_deps = [("PROFESSIONAL.SOFTWARE", "Frequently"),
                ("PROFESSIONAL.SOFTWARE", "Occasionally")]
dep_oss_hiring = [("PARTICIPATION.TYPE.CONTRIBUTE", 1)]
dep_min_home = [("IMMIGRATION", "Yes, and I intend to stay permanently."),
                ("IMMIGRATION", "Yes, and I intend to stay temporarily."),
                ("IMMIGRATION", "Yes, and I am not sure about my future plans.")]
dep_part_any = [("PARTICIPATION.TYPE.FOLLOW", 1), ("PARTICIPATION.TYPE.USE.APPLICATIONS", 1),
                ("PARTICIPATION.TYPE.USE.DEPENDENCIES", 1), ("PARTICIPATION.TYPE.CONTRIBUTE", 1),
                ("PARTICIPATION.TYPE.OTHER", 1)]
dep_off_site = [("POPULATION", "off site community")]


# Maps a column to its dependencies
cols_dependencies = [("CONTRIBUTOR.TYPE.CONTRIBUTE.CODE", dep_contribute),
                     ("CONTRIBUTOR.TYPE.CONTRIBUTE.DOCS", dep_contribute),
                     ("CONTRIBUTOR.TYPE.PROJECT.MAINTENANCE", dep_contribute),
                     ("CONTRIBUTOR.TYPE.FILE.BUGS", dep_contribute),
                     ("CONTRIBUTOR.TYPE.FEATURE.REQUESTS", dep_contribute),
                     ("CONTRIBUTOR.TYPE.COMMUNITY.ADMIN", dep_contribute),
                     ("PROFESSIONAL.SOFTWARE", dep_employed),
                     ("OSS.USER.PRIORITIES.LICENSE", dep_user_priors),
                     ("OSS.USER.PRIORITIES.CODE.OF.CONDUCT", dep_user_priors),
                     ("OSS.USER.PRIORITIES.CONTRIBUTING.GUIDE", dep_user_priors),
                     ("OSS.USER.PRIORITIES.CLA", dep_user_priors),
                     ("OSS.USER.PRIORITIES.ACTIVE.DEVELOPMENT", dep_user_priors),
                     ("OSS.USER.PRIORITIES.RESPONSIVE.MAINTAINERS", dep_user_priors),
                     ("OSS.USER.PRIORITIES.WELCOMING.COMMUNITY", dep_user_priors),
                     ("OSS.USER.PRIORITIES.WIDESPREAD.USE", dep_user_priors),
                     ("OSS.CONTRIBUTOR.PRIORITIES.LICENSE", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.CODE.OF.CONDUCT", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.CONTRIBUTING.GUIDE", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.CLA", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.ACTIVE.DEVELOPMENT", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.RESPONSIVE.MAINTAINERS", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.WELCOMING.COMMUNITY", dep_contribute),
                     ("OSS.CONTRIBUTOR.PRIORITIES.WIDESPREAD.USE", dep_contribute),
                     ("INFO.JOB", dep_info_job),
                     ("TRANSPARENCY.PRIVACY.PRACTICES.OSS", dep_contribute),
                     ("FIND.HELPER", dep_rec_help),
                     ("RECEIVED.HELP.TYPE", dep_rec_help),
                     ("HELPER.PRIOR.RELATIONSHIP", dep_rec_help),
                     ("FIND.HELPEE", dep_prov_help),
                     ("PROVIDED.HELP.TYPE", dep_prov_help),
                     ("HELPEE.PRIOR.RELATIONSHIP", dep_prov_help),
                     ("OSS.AS.JOB", dep_employed),
                     ("OSS.AS.JOB", dep_oss_job),
                     ("OSS.AT.WORK", dep_employed),
                     ("OSS.IP.POLICY", dep_employed),
                     ("EMPLOYER.POLICY.APPLICATIONS", dep_employed),
                     ("EMPLOYER.POLICY.DEPENDENCIES", dep_employed),
                     ("EMPLOYER.POLICY.DEPENDENCIES", dep_emp_deps),
                     ("OSS.HIRING", dep_employed), # check employed first
                     ("OSS.HIRING", dep_oss_hiring),
                     ("MINORITY.HOMECOUNTRY", dep_min_home),
                     ("PARTICIPATION.TYPE.ANY.REPONSE", dep_part_any),
                     ("OFF.SITE.ID", dep_off_site)]



# Check columns match expected format (date, int, bool or string)
def enforce_col_types(df):

    # coerce to dates
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%y %H:%M', errors='coerce')

    # coerce to ints
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # unnecessary to convert remaining columns as they are strings
    # which are still objects in pandas
    # for col in string_cols:
    #     df[col] = df[col].astype('str')
    print("Finished enforcing column types.")
    return df


# Ensures that only accepted values are present in the column of the
# dataframe passed in.
# Counts the number of values that do not match with the accepted option.
# Can be used with numbers by supplying the list of acceptable integers
def contains_acceptable_values(df, col_name):
    errors = []

    # if checking all columns, iterate through the mapping of column names to
    # their accepted values (defined above as cols_exp_values) and check each
    # contains only the expected values
    if (col_name == "all" or col_name == "ALL"):
        first = True
        for name, accepted in cols_exp_values:
            # check each column against the acceptable values
            col_errs = check_column(df, name, accepted)

            if len(col_errs) > 0:
                # safeguard against keeping the empty list as a first element
                if first:
                    errors = col_errs
                else:
                    errors.append(col_errs)

    # otherwise check for the specific col_name entered
    else:
        # find the list of accepted values for the column
        accepted = [pair[1] for pair in cols_exp_values if pair[0] == col_name]

        # if column is not recognized, print an error message and an error value
        if len(accepted) == 0:
            if (col_name == "RESPONSE.ID"):
                print("This column accepts any integer.")
                return [0]

            if (col_name == "DATE.SUBMITTED"):
                print("This column accepts any date.")
                return [0]

            else:
                print("Column name {} not recognized!".format(col_name))
                # not possible to have -1 incorrect values, so signals error
                return [-1]

        # otherwise extract the element from the singleton list
        accepted = accepted[0]
        # check the column against the acceptable values
        col_errs = check_column(df, name, accepted)
        # if any invalid values found, append them to the empty list of errors
        if len(col_errs) > 0:
            errors = append(col_errs)

    errorCount = 0
    if len(errors) == 0:
        print("Summary: {} errors across {}".format(errorCount, col_name))

    else:
        print("Invalid values encountered:")
        for val, count in errors:
            errorCount += count
            print("    {} occurrences of \"{}\"".format(count, val))

        print("Summary: {} errors across {}".format(errorCount, col_name))

# checks a specific column (col_name) in the dataframe (df) contains only
# admissable values (accepted)
def check_column(df, col_name, accepted):
    errors = []

    # iterate through the rows of the column indicated
    for index, rowData in df[col_name].iteritems():

        # if the entry at any row is not expected, print a message indicating
        # the location of the entry, the value of the entry and increment errors
        if (rowData not in accepted) and not (pd.isna(rowData)):

            found = False
            if len(errors) > 0:

                for item in errors:
                    if rowData == item[0]:
                        update = item[1] + 1
                        errors[errors.index(item)] = (item[0], update)
                        found = True

            if not found:
                new_err = (rowData, 1)
                errors.append(new_err)

    # return the inconsistent entries and their counts
    return errors


# used to clean dataframe or column depending on input
def clean_values(df, col_name):
    count = 0

    # if checking all columns, iterate through the mapping of column names to
    # their accepted values (defined above as cols_exp_values) and check each
    # contains only the expected values
    if (col_name == "all" or col_name == "ALL"):
        for name, accepted in cols_exp_values:
            # check each column against the acceptable values
            df[col_name] = clean_column(df, name, accepted)

    # otherwise check for the specific col_name entered
    else:
        # find the list of accepted values for the column
        accepted = [pair[1] for pair in cols_exp_values if pair[0] == col_name]

        # if column is not recognized, print an error message and an error value
        if len(accepted) == 0:
            if (col_name == "RESPONSE.ID"):
                print("Non-numerical values have already been marked as NaN in type enforcing.") # iterate here marking any non numbers as NaN
                return 0

            if (col_name == "DATE.SUBMITTED"):
                print("Non-date values have already been coerced in type enforcing.") # iterate here making any non
                return 0

            else:
                print("Column name {} not recognized!".format(col_name))
                # not possible to have -1 incorrect values, so signals error
                return -1

        # otherwise extract the element from the singleton list
        accepted = accepted[0]
        # check the column against the acceptable values
        df[col_name] = clean_column(df, col_name, accepted)

    print("Cleaning of {} has completed.".format(col_name))
    return df


# Used to clean unaccepted values from the data
def clean_column(df, col_name, accepted):
    # iterate through the rows of the column indicated
    for index, rowData in df[col_name].iteritems():

        # if the entry at any row is not expected, set it to NaN
        if (rowData not in accepted):
            df.iloc[index, df.columns.get_loc(col_name)] = np.nan

    return df[col_name]


# checks that hidden questions are not answered unless certain conditions met
def enforce_dependencies(df, col_name):
    changes = 0

    # check all columns with dependencies
    if (col_name == "all" or col_name == "ALL"):
        for name, conditions in cols_dependencies:
            # check each column against the acceptable values
            result = enforce_col_depends(df, name, conditions)

            # update the dataframe
            df = result[2]
            # count the changes made
            changes += result[1]
            # verify the condition was met at least once
            # if not result[0]:
            #     print("Condition not met for any row of", name)

    # otherwise check for the specific col_name entered
    else:
        # find the list of accepted values for the column
        conditions = [pair[1] for pair in cols_dependencies if pair[0] == col_name]

        # if column is not recognized, print an error message and an error value
        if len(conditions) == 0:
            print("Column name {} does not have any recognized dependencies.".format(col_name))
            return df

        # otherwise extract the element from the singleton list
        conditions = conditions[0]
        # check the column against the required conditions values
        result = enforce_col_depends(df, col_name, conditions)

        # update the dataframe
        df = result[2]
        # count the changes made
        changes += result[1]
        # verify the condition was met at least once
        # if not result[0]:
        #     print("Condition not met for any row of", col_name)

    print("Enforcing dependencies of {} has completed. {} changes have been made.".format(col_name, changes))
    return df


# checks a specific column meets the required dependencies:
# ie. Hidden unless: Response to # is one of the following (a or b)
def enforce_col_depends(df, name, conditions):
    changes = 0
    conditionMet = False

    # iterate through the rows of the dataframe
    for index, row in df.iterrows():

        # iterate through the lists of columns and required conditions
        for col, condition in conditions:

            # if encounter an ("AND", "AND") tuple, it means the following two
            # lists both need to have at least one condition met
            # if (col == "AND") and (condition == "AND"):
            #
            #     print("found AND")
            #     # check if at least one condition is met in the first list
            #     condition1 = enforce_col_depends(df, name, conditions[1])
            #     # check if at least one condition is met in the second list
            #     condition2 = enforce_col_depends(df, name, conditions[2])
            #
            #     # enforce that at least one condition in each list has been met
            #     conditionMet = (condition1[0] and condition2[0])
            #     changes = condition1[1] + condition2[1]
            #
            #     return (conditionMet, changes)

            # else:
            if (row[col] == condition):
                conditionMet = True # signal that at least one condition has been met
                continue

            else: # if already nan then question was correctly hidden
                if not pd.isna(df.at[index, name]):
                    # print(df.at[index, name], "-> NaN")
                    df.iloc[index, df.columns.get_loc(name)] = np.nan
                    changes += 1

            # print("after:", row[col])

    # return if the condition was met and the number of changes made
    return (conditionMet, changes, df)
