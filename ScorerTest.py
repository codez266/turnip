# -*- coding: utf-8 -*-
import unittest
from scorer import Scorer, CountScorer, SVMScorer
class ScorerTest(unittest.TestCase):
    def setUp(self):
        self.topics = ['Lawyer', 'Actor', 'Artist', 'Politician']
        self.mode = 'profession'
        self.pairs = [['Aamir Khan', 'Actor', 7],['Aamir Khan', 'Lawyer', 0],['Barack Obama', 'Lawyer', 4],[ 'Barack Obama', 'Artist', 0], ['Barack Obama', 'Politician', 7]]
        self.scorer = Scorer(self.topics, self.pairs, self.mode)
        self.indicators = {'Lawyer': ['law', 'lawyer', 'attorney', 'justice'],
                'Actor':['actor', 'acting', 'role', 'starred', 'cast'],
                'Artist':['artist', 'art', 'painting'],
                'Politician':['politician','political','politicis','congress']}

    def tearDown(self):
        #self.scorer.dispose()
        self.scorer = None

class SVMScorerTest(ScorerTest):
    def setUp(self):
        super().__init__()
        super().setUp()
        self.scorer = SVMScorer(self.topics, self.pairs, self.mode)
    
    def test_num_features(self):
        self.assertEqual(self.scorer.num_features, 6)

    
    def test_count(self):
        person = 'Barack Obama'
        article = '\"Barack\" and \"Obama\" redirect here. For his father, see Barack Obama Sr. For other uses of \"Barack\", see Barack (disambiguation). For other uses of \"Obama\", see Obama (disambiguation). Barack Hussein Obama II (US i\/b\u0259\u02c8r\u0251\u02d0k hu\u02d0\u02c8se\u026an o\u028a\u02c8b\u0251\u02d0m\u0259\/ b\u0259-RAHK hoo-SAYN oh-BAH-m\u0259; born August 4, 1961) is an American politician who is the 44th and current President of the United States. He is the first African American to be elected to office and the first president born outside the contiguous United States. Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School, where he was president of the Harvard Law Review. He was a community organizer in Chicago before earning his law degree. He worked as a civil rights attorney and taught constitutional law at the University of Chicago Law School from 1992 to 2004'
        words = ['lawyer', 'professor', 'politician', 'law']
        c = self.scorer.count(person, article, words)
        self.assertEqual(c, 6 / len(article))

    def test_feature0(self):
        person = 'Barack Obama'
        article = '\"Barack\" and \"Obama\" redirect here. For his father, see Barack Obama Sr. For other uses of \"Barack\", see Barack (disambiguation). For other uses of \"Obama\", see Obama (disambiguation). Barack Hussein Obama II (US i\/b\u0259\u02c8r\u0251\u02d0k hu\u02d0\u02c8se\u026an o\u028a\u02c8b\u0251\u02d0m\u0259\/ b\u0259-RAHK hoo-SAYN oh-BAH-m\u0259; born August 4, 1961) is an American politician who is the 44th and current President of the United States. He is the first African American to be elected to office and the first president born outside the contiguous United States. Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School, where he was president of the Harvard Law Review. He was a community organizer in Chicago before earning his law degree. He worked as a civil rights attorney and taught constitutional law at the University of Chicago Law School from 1992 to 2004'
        c = self.scorer.feature0(person, article, self.scorer.persons['Barack Obama']['profession'], self.indicators)
        self.assertEqual(c, [0, 0, 1])

    def test_feature1(self):
        text = "This article is about Napoleon I. For other uses, see Napoleon (disambiguation).      Napoleon Bonaparte (Napol\u00e9on Bonaparte; \/n\u0259\u02c8po\u028ali\u0259n, -\u02c8po\u028alj\u0259n\/; French:\u00a0[nap\u0254le\u0254\u0303 b\u0254napa\u0281t], Italian:\u00a0[napoleo\u014be b\u0254\u014baparte], born \"Napoleone di Buonaparte\" (Italian:\u00a0[napoleo\u014be dj bu\u0254\u014baparte]); 15 August 1769 \u2013 5 May 1821) was a French military and political leader who rose to prominence during the French Revolution and led several successful campaigns during the Revolutionary Wars. As Napoleon I, he was Emperor of the French from 1804 until 1814, and again in 1815. Napoleon dominated European and global affairs for more than a decade while leading France against a series of coalitions in the Napoleonic Wars. He won most of these wars and the vast majority of his battles, building a large empire that ruled over continental Europe before its final collapse in 1815. One of the greatest commanders in history, his wars and campaigns are studied at military schools worldwide. Napoleon's political and cultural legacy has ensured his status as one of the most celebrated and controversial leaders in human history. He was born in Corsica to a relatively modest family from the minor nobility. When the Revolution broke out in 1789, Napoleon was serving as an artillery officer in the French army. He attempted to capitalize quickly on the new political situation by returning to Corsica in hopes of starting a political career. After that venture failed, he came back to the military and rose rapidly through the ranks, ending up as commander of the Army of Italy after saving the governing Directory by suppressing a revolt from royalist insurgents. At age 26, he began his first military campaign against the Austrians and their Italian allies, scoring a series of decisive victories, conquering the Italian Peninsula in a year, and becoming a national hero. In 1798, he led a military expedition to Egypt that served as a springboard to political power. He engineered a coup in November 1799 and became First Consul of the Republic. His rising ambition inspired him to go further, and in 1804 he became the first Emperor of the French. Intractable differences with the British meant that the French were facing a Third Coalition by 1805"
        self.scorer.feature1('Barack Obama', text, self.scorer.persons['Barack Obama']['profession'], self.indicators)
        self.assertTrue(True)
    
    def test_feature2(self):
        categoryBarack = ["CS1 maint: Multiple names: authors list","Pages containing links to subscription-only content","All articles with dead external links","Articles with dead external links from June 2016","Pages using ISBN magic links","Wikipedia indefinitely move-protected pages","Wikipedia indefinitely semi-protected biographies of living people","Use American English from December 2014","All Wikipedia articles written in American English","Use mdy dates from November 2016","Articles including recorded pronunciations","Wikipedia articles scheduled for update tagging","Articles with dead external links from November 2016","Spoken articles","Articles with hAudio microformats","Official website different in Wikidata and Wikipedia","Articles with DMOZ links","Articles with Project Gutenberg links","Articles with Internet Archive links","AC with 16 elements","Wikipedia articles with VIAF identifiers","Wikipedia articles with LCCN identifiers","Wikipedia articles with ISNI identifiers","Wikipedia articles with GND identifiers","Wikipedia articles with SELIBR identifiers","Wikipedia articles with BNF identifiers","Wikipedia articles with MusicBrainz identifiers","Wikipedia articles with NLA identifiers","Wikipedia articles with SBN identifiers","Featured articles","Barack Obama","Obama family","1961 births","20th-century American writers","20th-century scholars","21st-century American politicians","21st-century American writers","21st-century scholars","African-American non-fiction writers","African-American people in Illinois politics","African-American United States presidential candidates","African-American United States Senators","American civil rights lawyers","American community activists","American legal scholars","American Nobel laureates","American people of English descent","American people of Irish descent","American people of Kenyan descent","American people of Luo descent","American political writers","Articles containing video clips","Columbia University alumni","Democratic Party Presidents of the United States","Democratic Party (United States) presidential nominees","Democratic Party United States Senators","Harvard Law School alumni","Illinois State Senators","Living people","Nobel Peace Prize laureates","Occidental College alumni","Politicians from Chicago","Politicians from Honolulu","Presidents of the United States","Progressivism in the United States","Punahou School alumni","United States presidential candidates, 2008","United States presidential candidates, 2012","United States Senators from Illinois","University of Chicago Law School faculty","Writers from Chicago"]
        score = self.scorer.feature2('Barack Obama', categoryBarack, self.scorer.persons['Barack Obama']['profession'], self.indicators)
        self.assertEquals(score, [1, 0, 1])
        categoryAmir = ["Pages containing links to subscription-only content","Webarchive template wayback links","Wikipedia indefinitely semi-protected biographies of living people","EngvarB from March 2014","Use dmy dates from March 2014","Pages using deprecated image syntax","Biography with signature","Articles with hCards","Commons category with local link same as on Wikidata","Wikipedia articles with VIAF identifiers","Wikipedia articles with LCCN identifiers","Wikipedia articles with ISNI identifiers","Wikipedia articles with GND identifiers","Wikipedia articles with BNF identifiers","Wikipedia articles with MusicBrainz identifiers","Wikipedia articles with NLA identifiers","Aamir Khan","1965 births","Living people","People from Mumbai","People from Hardoi","Indian Muslims","Indian male voice actors","Indian male film actors","Indian male film producers","Male actors in Hindi cinema","Hindi-language film directors","Indian male film directors","Recipients of the Padma Shri in arts","Recipients of the Padma Bhushan in arts","Indian male child actors","Indian film actors working since their childhood","20th-century Indian film directors"]
        score = self.scorer.feature2('Aamir Khan', categoryAmir, self.scorer.persons['Aamir Khan']['profession'], self.indicators)
        self.assertEquals(score, [1, 0])
    
    def test_features_gen(self):
        self.scorer.persons['Aamir Khan']['features'] = [[1,0],[0,1],[3,1],[2,3],[1,4],[3,2]]
        self.scorer.persons['Barack Obama']['features'] = [[1,1],[2,1],[3,2],[1,3],[3,2],[2,3]]
        features = self.scorer.genFeatures()
        line = '{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} # {}\n'
        s = ""
        s = s + line.format(7, 1, 1, 0, 3, 2, 1, 3, "Aamir Khan")
        s = s + line.format(0, 1, 0, 1, 1, 3, 4, 2, "Aamir Khan")
        s = s + "#\n"
        s = s + line.format(4, 2, 1, 2, 3, 1, 3, 2, "Barack Obama")
        s = s + line.format(0, 2, 1, 1, 2, 3, 2, 3, "Barack Obama")
        s = s + "#\n"
        self.assertEquals(features, s)

