import 'package:flutter/material.dart';

class EncyclopediaPage extends StatelessWidget {
  const EncyclopediaPage({super.key});

  final List<Map<String, String>> diseases = const [
    {
      "acronym": "MEL",
      "name": "Melanom",
      "risk": "Çok Yüksek Risk",
      "description": "Cilt kanserlerinin en tehlikeli türüdür. Renkleri genellikle siyah, koyu kahverengi veya mavi/kırmızı karışımıdır. Sınırları düzensizdir ve zamanla büyür. Erken teşhis hayat kurtarır, mutlaka acil olarak Dermotoloğa başvurulmalıdır.",
      "color": "redAccent"
    },
    {
      "acronym": "BCC",
      "name": "Bazal Hücreli Karsinom",
      "risk": "Yüksek Risk",
      "description": "En sık görülen cilt kanseridir. Güneşe maruz kalan bölgelerde pembe renkli, şeffafımsı veya üzerinde ince damarlar olan küçük yumrular şeklinde çıkar. Yavaş büyür ancak mutlaka tedavi edilmelidir.",
      "color": "redAccent"
    },
    {
      "acronym": "AKIEC",
      "name": "Aktinik Keratoz",
      "risk": "Yüksek Risk (Öncül)",
      "description": "Kanser değildir ancak kansere (Skuamöz Hücreli Karsinom) dönüşme riski çok yüksektir. Güneş hasarı olan bölgelerde kuru, pullu ve sert döküntüler şeklindedir.",
      "color": "redAccent"
    },
    {
      "acronym": "VASC",
      "name": "Vasküler Lezyonlar",
      "risk": "Orta Risk",
      "description": "Kan damarlarının cilt yüzeyinde birikmesiyle oluşan kırmızı/mor renkli lezyonlardır (Kiraz Anjiomu vb.). Çoğu zaman iyi huyludur ancak kanama yapıyorsa veya aniden büyüdüyse incelenmelidir.",
      "color": "orangeAccent"
    },
    {
      "acronym": "NV",
      "name": "Melanositik Nevüs",
      "risk": "Düşük Risk",
      "description": "Neredeyse her insanda görülen klasik, iyi huylu, standart 'Ben'lerdir. Sınırları net, rengi genelde tek tonlu kahverengidir ve zararsızdır.",
      "color": "greenAccent"
    },
    {
      "acronym": "BKL",
      "name": "Benign Keratoz",
      "risk": "Düşük Risk",
      "description": "Yaş ilerledikçe ortaya çıkan 'yaşlılık lekeleri' (Seboreik Keratoz) dir. Cilde sonradan hamur/sakız yapıştırılmış gibi duran, pürüzlü ve kabuklu ama tamamen iyi huylu oluşumlardır.",
      "color": "greenAccent"
    },
    {
      "acronym": "DF",
      "name": "Dermatofibrom",
      "risk": "Düşük Risk",
      "description": "Çoğunlukla bacak veya kollarınızda böcek ısırığı veya hafif bir travma sonrası cilt altında kalan sert bir lif/nedbe dokusudur. İyi huyludur.",
      "color": "greenAccent"
    }
  ];

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: diseases.length,
      itemBuilder: (context, index) {
        final d = diseases[index];
        final riskColor = d['color'] == 'redAccent' ? Colors.redAccent 
                        : d['color'] == 'orangeAccent' ? Colors.orangeAccent 
                        : const Color(0xFF00E5FF);

        return Card(
          elevation: 4,
          shadowColor: const Color(0xFF00E5FF).withOpacity(0.05),
          color: const Color(0xFF131B2F),
          margin: const EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: BorderSide(color: const Color(0xFF00E5FF).withOpacity(0.15), width: 1)
          ),
          child: ExpansionTile(
            iconColor: const Color(0xFF00E5FF),
            collapsedIconColor: Colors.white70,
            leading: CircleAvatar(
              backgroundColor: riskColor.withOpacity(0.15),
              child: Text(d['acronym']!, style: TextStyle(color: riskColor, fontWeight: FontWeight.bold, fontSize: 12)),
            ),
            title: Text(d['name']!, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
            subtitle: Text(d['risk']!, style: TextStyle(color: riskColor, fontWeight: FontWeight.w600)),
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  d['description']!,
                  style: const TextStyle(fontSize: 15, height: 1.5, color: Colors.white70),
                ),
              )
            ],
          ),
        );
      },
    );
  }
}
